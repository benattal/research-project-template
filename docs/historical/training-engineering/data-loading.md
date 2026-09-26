# Data loading

The data pipeline here did more than read files: datasets drew fresh noise
for each sample every epoch, split measurements into views, and wrapped a
virtual epoch length around a real image count. That per-epoch state
interacts with DataLoader worker processes, and a loader option added for
speed changed what the model trained on. These entries cover worker
processes and host-to-device copies, the CPU cost of preparing items, and
mapping indices and labels onto the files on disk.

## Workers and copies

### Persistent workers freeze per-epoch dataset state

Do not set `persistent_workers=True` when the dataset holds state that
changes between epochs, such as an epoch counter that seeds per-sample
noise.

**Why.** DataLoader workers are separate processes, each holding a copy of
the dataset made when the workers start. Without persistent workers, every
epoch's iterator starts fresh workers from the main process's current
dataset, so `dataset.set_epoch(epoch)` reaches them. With persistent
workers, the copies from the first epoch live for the whole run; the epoch
counter they read never advances, and the "fresh" noise repeats every epoch.

**How to apply.** Leave the option off and say why at the call site:

```python
loader = DataLoader(
    dataset, batch_size=per_process_batch, shuffle=True, drop_last=True,
    num_workers=cfg.num_workers, pin_memory=True,
    # Per-sample noise is keyed by dataset.epoch. Persistent worker copies
    # would keep their first epoch forever and freeze the noise.
    persistent_workers=False)
```

To keep persistent workers, pass the epoch through the sampler instead (a
sampler that yields `(epoch, index)` pairs), so the state travels with the
indices.

**Evidence.** A speed change added `persistent_workers=True`,
`prefetch_factor=4` and `pin_memory=True` together and cut data-loading time
per step by 4 to 13 ms. Afterward, a model still learned the right
conditional mean, but its individual samples carried high-frequency noise
that grew worse through epoch 200 and beyond. Bisection isolated that change;
removing the three loader options restored the earlier loss trajectory epoch
for epoch and produced sample grids indistinguishable from the earlier ones.

### Pin host memory when copies are marked `non_blocking`

`tensor.to(device, non_blocking=True)` from pageable host memory is a
synchronous copy. Set `pin_memory=True` on the DataLoader so that the
non-blocking copies overlap with compute.

**Evidence.** After the options above were removed, the loaders copied 30 to
40 MB per step per GPU from pageable memory, and every `non_blocking=True`
in the input path had no effect. `pin_memory=True` was restored on its own;
it holds no per-epoch state.

### Run main-process dataset loops with one intra-op thread

Wrap loops that call `dataset[i]` in the main process, such as calibrating a
normalization constant from the first thousand items, in a context that sets
`torch.set_num_threads(1)`.

**Why.** The main process runs torch operations on an intra-op thread pool
sized to the machine's cores. A dataset item built from many small tensor
operations spends its time waking and synchronizing that pool. PyTorch sets
one thread in DataLoader workers, so the same code is fast inside a loader
and slow outside it.

**How to apply.**

```python
@contextmanager
def single_threaded_torch():
    n = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        yield
    finally:
        torch.set_num_threads(n)

with single_threaded_torch():
    items = [dataset[i] for i in range(1024)]
```

**Evidence.** One item took 13.8 ms in the main process with the default
pool and 1.44 ms with one thread. A calibration over 1024 items took about
14 s instead of 1.5 s, on every process at every job start.

### Cache decoded and resized images when decoding dominates

When items are JPEG files decoded, cropped and resized every epoch, cache the
resized images once as a uint8 array (a memory-mapped file shares one copy
across workers through the page cache).

**Why.** Decoding and a bicubic resize took 0.55 ms per item, against
0.008 ms to read a resized image from the cache, 40 to 55% of the CPU cost
of an item. The cache for a 64×64 split of about 150k images was 1.9 GB.

**How to apply.** Caching is safe for determinism only when loading is a
pure function of the index, with all randomness applied after the load.
Rounding the resized image to uint8 changes values by up to 0.5/255 per
pixel, so cached and uncached runs are not bit-identical: keep the cache
behind an explicit setting and never switch it in the middle of a run.

### Do not ship per-item constant tensors

An index tensor that was the same for every item (an identity index over
the 4096 pixels of each of four views, 128 KB of int64 per item) was built,
collated and copied to the GPU with every item: 8 to 16 MB per step of
redundant host-to-device traffic.
Detect such constant structure once, from the configuration or the first
batch, and use one shared tensor.

## Indices and labels

### A virtual epoch shorter than the dataset must rotate through it

When the configured epoch length is smaller than the number of real items
and the dataset maps indices with `real = index % n_real`, every epoch serves
the same prefix and the rest of the data is never seen. Offset the mapping
by the epoch:

```python
real = (epoch * epoch_length + index) % n_real
```

**Evidence.** Two families of configurations with a virtual epoch of 50,000
items trained on the first 50,000 of about 150,000 images for their whole
run, while a comment on one of them described training on the full set.

### Key labels by file name, not by row order

Look labels up by file name, generate the label file from the same pass over
the source that extracts the images, and write labels only for images that
reached the disk. A separately obtained label file whose rows were offset
from the extracted images silently mislabeled every conditioned sample.

`torchvision.datasets.ImageFolder` assigns labels by position in the sorted
list of class folders present, so a split with a missing class folder
renumbers every later class. When class folders are named by integer
labels, parse the names instead.

### Record the real item count after a download

Hosted dataset shards can be corrupt or short. Record the number of items
that reached the disk and use it wherever a configuration assumes the full
count. In one download, 11 of 132 parquet shards were unreadable and about 8%
of the images were lost.
