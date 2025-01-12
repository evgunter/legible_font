This is a project to create a character set which is most easily distinguishable at various scales.
In `base_boards.rs` is the code to generate shift- and scale-invariant "characters" of 3 x 3 binary pixels.
We will constrain the full characters to be coarse-grained to these fundamental characters, so that the characters are distinguishable at the smallest possible scale.