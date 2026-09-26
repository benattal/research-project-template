# Architecture tests

Static ownership, naming, source-contract, placement, and hot-path guards.
They read files and git state only: they never import an application entry
point or initialize a GPU, so they run anywhere, including login nodes.
