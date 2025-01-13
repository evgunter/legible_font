### Overall
This is a project to create a character set which is most easily distinguishable at various scales.

### Smallest-resolution characters
In `min_res` is the code to generate shift- and scale-invariant "characters" of 3 x 3 binary pixels.
We will constrain the full characters to be coarse-grained to these fundamental characters, so that the characters are distinguishable at the smallest possible scale.

### Scaling to greater resolutions
In `upscale` will be the code to generate highly distinct character shapes which are appropriately rasterized to their minimum-resolution versions.
We will do so by using a pretrained vision model to embed the candidate characters, and then optimize such that the embeddings are far away from each other.
In particular, we will attempt to minimize the "energy" of the set of characters $\{c_i\}$:
$\sum_{i < j} \|\text{embed}(c_i) - \text{embed}(c_j)\|^{-1}$.
