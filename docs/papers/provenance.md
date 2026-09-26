# Research provenance and retained decisions

This page retains the implementation decisions that remain binding after the
investigations that produced them were archived. Live experiment state belongs
in `docs/tracking/`; intended work belongs in `docs/plans/`; stable
engineering contracts belong in the focused guides and concepts.

Promote a finding here when a plan closes: state the decision, the evidence
that supports it (the control, its numbers, and where its artifacts live), and
the contract it imposes on code. Group entries by subject, not by date.

TODO(template): add the project's first retained decisions.

## Example: evaluation peak

- PSNR for the example application uses the peak-to-peak range of the
  reference function on the evaluation interval, fixed in code. A peak taken
  from each run's maximum made arms incomparable.
