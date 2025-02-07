mod base_boards;
use crate::base_boards::{Board, N, M};

// to export, use `cargo run > ../test_data/boards.txt`
// to display graphically, use `cargo run --features human_readable`

#[cfg(feature = "human_readable")]
const HUMAN_READABLE: bool = true;

#[cfg(not(feature = "human_readable"))]
const HUMAN_READABLE: bool = false;

fn main() {
    // run get_boards on a 3x3 board of bools
    let boards = i16::get_boards();
    if HUMAN_READABLE {
        print!("{} {}x{} boards\n\n", boards.len(), N, M);
        for board in boards {
            board.print();
            for _ in 0..2*M-1 {
                print!("-");
            }
            println!();
        }
    } else {
        println!("[{}, {}]", N, M);
        println!("{:?}", boards);
    }
}
