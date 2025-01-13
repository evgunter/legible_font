// to create a character set which is easily distinguishable, we will use a pretrained vision model to embed
// the candidate characters, and then optimize such that the embeddings are far away from each other.
// in particular, we will find the configuration of characters c_i which minimizes the "energy"
// sum_{i < j} 1/||embed(c_i) - embed(c_j)||


use embed_anything::{embed_image_directory, embeddings::embed::Embedder};
use std::error::Error;
use std::path::PathBuf;
use std::sync::Arc;

#[cfg(test)]
use tokio;

fn dist(v1: &Vec<f32>, v2: &Vec<f32>) -> f32 {
    // TODO: this should be replaced with the native norm for the tensor type of the ML library we use
    v1.iter()
      .zip(v2.iter())
      .map(|(a, b)| (a - b).powi(2))
      .sum::<f32>().sqrt()
}

fn loss(embeddings: &Vec<Vec<f32>>) -> Result<f32, Box<dyn Error>> {
    let energy = (0..embeddings.len()).flat_map(|i| (0..i).map(move |j| dist(&embeddings[i], &embeddings[j])))
                                      .map(|distance| 1.0 / distance)
                                      .sum::<f32>();
    Ok(energy)
}

#[test]
fn test_embed() {  // this runs a forward pass of CLIP on 3 images, so is very slow (~1min on my CPU)
    let manifest_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let project_root = match manifest_dir.parent() {
        Some(par) => par,
        None => panic!("cargo manifest dir does not have a parent directory")
    };
    
    let test_data_dir = project_root.join("test_data");

    println!("embedding images in {:?}", test_data_dir);
    
    let embedder = Arc::new(Embedder::from_pretrained_hf("clip", "openai/clip-vit-base-patch16", None).unwrap());
    
    let rt = tokio::runtime::Runtime::new().unwrap();
    let embeddings_w_meta = match rt.block_on(embed_image_directory(test_data_dir, &embedder, None, None::<fn(Vec::<embed_anything::embeddings::embed::EmbedData>)>)).unwrap() {
        Some(embs) => embs,
        None => panic!("embed_image_directory did not return any embeddings"),
    };

    // raise an error if there are no embeddings
    if embeddings_w_meta.is_empty() {
        panic!("no embeddings found");
    }

    // let embeddings = embeddings_w_meta.clone().into_iter().map(|e_w_m| (e_w_m.metadata.map(|m| m["file_name"]), e_w_m.embedding.to_dense().map_err(|e| e.into()))).collect::<Result<Vec<(Option<String>, Vec<f32>)>, Box<dyn Error>>>().unwrap();
    let embeddings = embeddings_w_meta.into_iter().map(|emb| {
        let filename = emb.metadata
            .and_then(|m| m.get("file_name").cloned())
            .ok_or("Missing file_name in metadata")?;
        
        let dense = emb.embedding.to_dense()?;
        
        Ok((filename, dense))
    })
    .collect::<Result<Vec<(String, Vec<f32>)>, Box<dyn Error>>>().unwrap();

    // check that the filenames are as we expect: img1.jpg, img1_modified.jpg, img2.jpg
    let filenames = {
        let mut flnms = embeddings.iter().map(|(f, _)| f).collect::<Vec<&String>>();
        flnms.sort();
        flnms
    };

    let expected_filenames = vec!["img1.jpg", "img1_modified.jpg", "img2.jpg"].into_iter().map(|s| project_root.join("test_data").join(s).to_str().unwrap().to_string()).collect::<Vec<String>>();
    assert_eq!(filenames, expected_filenames.iter().map(|s| s).collect::<Vec<&String>>());

    // get the embeddings for each of the expected filenames
    let img1 = &embeddings.iter().find(|(f, _)| f == &expected_filenames[0]).unwrap().1;
    let img1_modified = &embeddings.iter().find(|(f, _)| f == &expected_filenames[1]).unwrap().1;
    let img2 = &embeddings.iter().find(|(f, _)| f == &expected_filenames[2]).unwrap().1;

    // check that img1 and img1_modified are at least 50% closer to each other than to img2
    let dist_img1_img1_modified = dist(&img1, &img1_modified);
    let dist_img1_img2 = dist(&img1, &img2);
    let dist_img1_modified_img2 = dist(&img1_modified, &img2);

    assert!(dist_img1_img1_modified < dist_img1_img2 / 2.0);
    assert!(dist_img1_img1_modified < dist_img1_modified_img2 / 2.0);
}


    // TODO move to where needed
    // let distances = (0..embeddings.len()).flat_map(|i| (0..i).map(|j| {
    //     let ei = &embeddings[i];
    //     let ej = &embeddings[j];
    //     (ei.0.clone(), ej.0.clone(), dist(&ei.1, &ej.1))
    // }).collect::<Vec<(String, String, f32)>>()).collect::<Vec<(String, String, f32)>>();


    // TODO move to where needed
    // let emb_values = embeddings.clone().into_iter().map(|(_, v)| v).collect();
    // let energy = loss(&emb_values).unwrap();
    // println!("total energy: {}", energy);

    // let out_dir = project_root.join("test_data_out");

    // println!("writing embeddings to {:?}", out_dir);

    // // write the embeddings to files
    // for (i, embedding) in embeddings.iter().enumerate() {
    //     let path = out_dir.join(format!("{}.npy", i));
    //     // convert the embedding to a string
    //     // TODO: this is a placeholder; i'm not planning on reading stuff from here, just glancing at it
    //     let data = format!("{:?}", embedding);
    //     // write the string to a file
    //     std::fs::write(&path, data).unwrap();
    // }


fn main() {
    println!("nothing to do");
}
