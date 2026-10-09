# Separate the toolkit from the paper repositories

The reusable inverse landscape genetics package will live in a new `ilg-toolkit` repository, with Wade-specific data handling and paper reproduction workflows retained in their existing research repositories. This trades some shared maintenance for a clear boundary between a data-agnostic tool and fixed scientific studies, so toolkit APIs and defaults can evolve independently of paper reproduction.
