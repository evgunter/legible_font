// generate all the N x M pixel configurations
// such that the nonempty regions are not shifts of each other:
// x _ _     _ x _     _ _ _
// _ x _  =  _ _ x  =  _ x _
// _ _ _     _ _ _     _ _ x
// (in practice implemented by ensuring that the first row and column are nonempty)
//
// and the characters are size-invariant:
// x x x     x x _     x x x     x _ _
// x _ _  =  x _ _  ;  x x x  =  _ _ _
// x _ _     _ _ _     x x x     _ _ _
// (in practice implemented by ensuring that the shape does not fit in a 2 x 2 square)

pub const N: usize = 3;
pub const M: usize = 3;

pub trait Pixel: Eq + Sized {}  // a single pixel; can be b/w, grayscale, rgb, etc

pub trait Board<P: Pixel>: Sized {
    // a board of pixels, representing a letter at a certain size
    fn to_matrix(&self) -> [[P; M]; N];  // convert from the internal representation to a matrix
    fn get_boards() -> Vec<Self>;  // get all possible boards
    fn print(&self) -> ();
}

impl Pixel for bool {}

impl Board<bool> for i16 {
    fn to_matrix(&self) -> [[bool; M]; N] {
        // the board is represented by the least significant N*M bits of a 16 bit integer,
        // where the least significant bit is the top left corner of the board, the next bit is the second element of the top row, etc
        let mut matrix = [[false; M]; N];
        let mut n = *self;
        for i in 0..N {
            for j in 0..M {
                matrix[i][j] = n & 1 == 1;
                n >>= 1;
            }
        }
        matrix
    }

    fn get_boards() -> Vec<i16> {
        // we'll generate boards by counting an i16 from 1 to 2^(N*M) - 1
        // to implement shift-invariance, we will ensure that the pattern is as close to the top left corner as possible;
        // if there is an empty left column or top row, we will skip the board
        // (since the version closer to the top left is the representative of that equivalence class)
        let mut boards = Vec::new();
        'board: for b in 0..(1 << (N * M)) {
            let mat = b.to_matrix();

            // ensure shift-invariance: check that the character cannot be moved further to the top left:
            'firstcol: for i in 0..N {
                if mat[i][0] {
                    break 'firstcol;
                }
                continue 'board;
            }
            'firstrow: for j in 0..M {
                if mat[0][j] {
                    break 'firstrow;
                }
                continue 'board;
            }

            // ensure scale-invariance: check that the character does not fit in a (N-1) x (M-1) square:
            // this works because any board that does not fit into a (N-1) x (M-1) square cannot be a 
            // scaled-down version of another board that fits into a N x M square, so we don't double-count.
            // we may miss some smaller boards which don't have a larger version, e.g.
            // x _ _
            // x x _  has no obvious 4 x 4 equivalent;
            // _ x x 
            // but this is ok for now, and it doesn't happen for N = M = 3
            // because the possible (already shift-invariant) 2 x 2 boards are:
            //  x    |  x x  |  x    |  x x  |  x    |  x x  |  x    |  x x
            //       |       |  x    |  x    |    x  |    x  |  x x  |  x x
            // all of which have 3 x 3 versions:
            // x x x | x x x | x     | x x x | x     | x x x | x     | x x x
            // x x x |       | x     | x     |   x   |     x | x     | x x x
            // x x x |       | x     | x     |     x |     x | x x x | x x x

            // we already checked that the first row and column are occupied,
            // so to check that it doesn't fit in an (N-1) x (M-1) square,
            // it suffices to check that one of the last row and column is occupied
            'scaleinvariance: {
                for i in 0..N {
                    if mat[i][M-1] {
                        break 'scaleinvariance;
                    }
                }
                for j in 0..M {
                    if mat[N-1][j] {
                        break 'scaleinvariance;
                    }
                }
                continue 'board;
            }
            boards.push(b);
        }
        boards
    }

    fn print(&self) {
        let matrix = self.to_matrix();
        for row in matrix.iter() {
            for pixel in row.iter() {
                print!("{} ", if *pixel { "x" } else { " " });
            }
            println!();
        }
    }
    
}

