mod base_boards;
use crate::base_boards::{Board, M};

fn main() {
    // run get_boards on a 3x3 board of bools
    let boards = i16::get_boards();
    print!("{} boards\n\n", boards.len());
    for board in boards {
        board.print();
        for _ in 0..2*M-1 {
            print!("-");
        }
        println!();
    }
}
