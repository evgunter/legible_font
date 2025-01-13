use anyhow::Result;
use candle_core::{DType, Device, Tensor, Module};
use candle_nn::{VarBuilder, VarMap, Optimizer, SGD};
use candle_transformers::models::dinov2::DinoVisionTransformer;
use rand::Rng;
use hf_hub::{api::sync::Api, Repo, RepoType};
use std::path::PathBuf;

// for writing bitmap images
use image::{RgbImage, Rgb, ImageFormat};

struct OptimizableImages {
    varmap: VarMap,
    // images: Vec<Tensor>,  // checking if just varmap suffices
    model: DinoVisionTransformer,
    optimizer: SGD,
}

// const IMAGE_DIMS: (usize, usize, usize) = (3, 224, 224);
const IMAGE_DIMS: (usize, usize, usize) = (3, 14, 14);  // TODO put back
const LEARNING_RATE: f64 = 0.01;

fn save_data_to_image(data: &Vec<f32>, filename: &str) -> Result<(), anyhow::Error> {
    let manifest_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let project_root = match manifest_dir.parent() {
        Some(par) => par,
        None => panic!("cargo manifest dir does not have a parent directory")
    };
    let test_data_dir = project_root.join("test_data_out");
    let path = test_data_dir.join(filename);

    let (_, d2, d3) = IMAGE_DIMS;
    let mut img = RgbImage::new(d2 as u32, d3 as u32);
    for x in 0..d2 {
        for y in 0..d3 {
            let rgb = Rgb([(255.0*data[x*d3*3+y*3+0]).round() as u8,  // TODO make less verbose
                          (255.0*data[x*d3*3+y*3+1]).round() as u8,
                          (255.0*data[x*d3*3+y*3+2]).round() as u8]);
            img.put_pixel(x as u32, y as u32, rgb);
        }
    }
    Ok(img.save_with_format(path, ImageFormat::Png)?)
}

fn save_tensor_to_image(img: &Tensor, filename: &str) -> Result<(), anyhow::Error> {
    let (d1, d2, d3) = IMAGE_DIMS;
    save_data_to_image(&img.reshape(d1*d2*d3)?.to_vec1()?, filename)
}

impl OptimizableImages {
    fn new(num_images: usize, device: &Device) -> Result<Self> {
        let mut varmap = VarMap::new();

        // Initialize random images
        // let mut images = Vec::with_capacity(num_images);
        let mut image_data = Vec::with_capacity(num_images);  // TODO remove
        let mut rng = rand::thread_rng();

        println!("Initializing images");  // TODO remove
        
        for i in 0..num_images {
            // let data: Vec<f32> = (0..3*224*224)
            //     .map(|_| rng.gen::<f32>())
            //     .collect();
            // let image = Tensor::from_vec(data, (3, 224, 224), device)?;
            // TODO: changing this to try to debug shape mismatch
            let data: Vec<f32> = (0..{
                let (d1, d2, d3) = IMAGE_DIMS;
                d1*d2*d3
            })
                .map(|_| rng.gen::<f32>())
                .collect();
            image_data.push(data.clone());   // TODO remove
            let image = Tensor::from_vec(data, IMAGE_DIMS, device)?;
            let img_name = format!("image{}", i);
            varmap.get(IMAGE_DIMS, &img_name, candle_nn::Init::Randn { mean: 0.0, stdev: 1.0 }, DType::F32, device)?;
            varmap.set_one(&img_name, image)?;
            println!("{} initialized", &img_name);  // TODO remove
        }
        
        println!("vmap (len {}): {:?}", varmap.all_vars().len(), varmap.all_vars());  // TODO remove

        // TODO remove this block
        // save all the images to a directory
        // currently the images are just 1-d vectors of pixel values. they need to be converted to a bitmap
        for (i, data) in image_data.iter().enumerate() {
            save_data_to_image(data, &format!("img{}.jpg", i))?;
        }




        // Load pretrained weights
        let api = Api::new()?;
        let repo = api.repo(Repo::with_revision(
            // "facebook/dinov2-small".to_string(),
            "lmz/candle-dino-v2".to_string(),
            RepoType::Model,
            "main".to_string(),
        ));
        
        let weights_filename = repo.get("dinov2_vits14.safetensors")?;
        
        println!("Weights path: {}", weights_filename.display());  // TODO remove
        
        let vb = unsafe {
            VarBuilder::from_mmaped_safetensors(&[weights_filename], DType::F32, device)?
        };
        
        println!("Model loaded");  // TODO remove
        
        // Create model with pretrained weights
        let model = DinoVisionTransformer::new(
            vb,
            12,   // depth for ViT-S
            384,  // embedding dimension for ViT-S
            6,    // attention heads for ViT-S
        )?;

        let optimizer = SGD::new(varmap.all_vars(), LEARNING_RATE)?;

        println!("Model created");  // TODO remove

        Ok(Self {
            varmap,
            // images,
            model,
            optimizer,
        })
    }

    fn get_embeddings(&self) -> Result<Vec<Tensor>> {
        let mut embeddings = Vec::with_capacity(self.varmap.all_vars().len());
        println!("Getting {} embeddings", self.varmap.all_vars().len());  // TODO remove
        
        for image in &self.varmap.all_vars() {
            // TODO: do i need to normalize?
            // let rescaled = (image.as_tensor() / 255.)?;
            let rescaled = image.as_tensor().div(&Tensor::full::<f32, (usize, usize, usize)>(255., IMAGE_DIMS, image.device())?)?;
            let channel_means: &[f32] = &[0.485, 0.456, 0.406];
            let channel_stds: &[f32] = &[0.229, 0.224, 0.225];
            let mean = Tensor::new(channel_means, image.device())?.reshape((3, 1, 1))?;
            let std = Tensor::new(channel_stds, image.device())?.reshape((3, 1, 1))?;
            let normalized = ((rescaled.broadcast_sub(&mean))?.broadcast_div(&std))?.unsqueeze(0)?;
            // let mean = Tensor::new(channel_means, image.device())?
            //     .reshape((3, 1, 1))?.broadcast_as(IMAGE_DIMS)?;
            // let std = Tensor::new(channel_stds, image.device())?
            //     .reshape((3, 1, 1))?.broadcast_as(IMAGE_DIMS)?;
            // let normalized = ((rescaled - mean)? / std)?;

            // Using get_intermediate_layers instead of forward gets us the correct embeddings, but it may be causing problems with the gradients
            // let embedding = self.model.get_intermediate_layers(
            //     &normalized,
            //     &[11],  // Get output from last layer
            //     true,   // reshape
            //     false, //true,   // return class token
            //     true,   // apply normalization
            // )?;
            // println!("Embedding gil: {:?}", embedding.squeeze(0)?.squeeze(0)?.squeeze(1)?.squeeze(1)?.to_vec1::<f32>()?);  // TODO remove
            // TODO: this returns logits rather than embeddings, but that might be useful for debugging whether get_intermediate_layers is the source of my problems
            let embedding = self.model.forward(&normalized)?;
            println!("Embedding fwd: {:?}", embedding.squeeze(0)?.to_vec1::<f32>()?);  // TODO remove
            
            println!("Embedding shape: {:?}", embedding.shape());  // TODO remove

            // TODO: embedding shape is currently [1, 1, 384, 1, 1]. can probably squeeze extra dimensions
            embeddings.push(embedding);
        }
        
        Ok(embeddings)
    }

    // fn loss(embeddings: &Vec<Vec<f32>>) -> Result<f32, Box<dyn Error>> {
    //     let energy = (0..embeddings.len()).flat_map(|i| (0..i).map(move |j| dist(&embeddings[i], &embeddings[j])))
    //                                       .map(|distance| 1.0 / distance)
    //                                       .sum::<f32>();
    //     Ok(energy)
    // }

    fn calculate_loss(&self) -> Result<Tensor> {
        let embeddings = self.get_embeddings()?;
        println!("embeddings got");  // TODO remove
        let mut loss = Tensor::zeros((), DType::F32, embeddings[0].device())?;
        
        for i in 0..embeddings.len() {
            for j in (i+1)..embeddings.len() {
                // let diff = (&embeddings[i] - &embeddings[j])?;
                let diff = &embeddings[i].sub(&embeddings[j])?;
                let norm = diff.sqr()?.sum_all()?.sqrt()?;
                // loss = (&loss + 1.0 / norm)?;
                loss = loss.add(&norm.recip()?)?;
            }
        }
        
        Ok(loss)

        // TODO: use the more functional style as in fn loss
        // let embeddings = self.get_embeddings()?;
        // let energy = (0..embeddings.len()).flat_map(|i| {
        //                                             let emb_copy = embeddings.clone();
        //                                             (0..i).map(move |j| {
        //                                                 let ei = &emb_copy[i];
        //                                                 let ej = &emb_copy[j];
        //                                                 let diff = (ei - ej)?;
        //                                                 let norm = diff.sqr()?.sum_all()?.sqrt()?;
        //                                                 1.0 / norm
        //                                             })
        //                                         }).reduce(|a, b| a? + b?).unwrap()?;
        // Ok(energy)
    }

    // TODO float datatype?
    fn optimization_step(&mut self) -> Result<f32, anyhow::Error> {
        println!("Optimization step");  // TODO remove
        let loss = self.calculate_loss()?;
        println!("Loss calculated: {:?}", loss);  // TODO remove
        let grads = loss.backward()?;
        // TODO: debugging why backward doesn't give us grads for the images

        println!("Gradients calculated: {:?}", grads);  // TODO remove

        self.optimizer.step(&grads)?;
        // TODO remove: pasting definition of .step from source code to debug
        // println!("varmap: {:?}", self.varmap.all_vars());  // TODO remove
        // for var in self.varmap.all_vars().iter() {
        //     if let Some(grad) = grads.get(var) {
        //         // print the var before and after
        //         println!("var before: {:?}", var);  // TODO remove
        //         var.set(&var.sub(&(grad * LEARNING_RATE)?)?)?;
        //         println!("var after: {:?}", var);  // TODO remove
        //     }
        // }
    
        println!("Optimizer step done");  // TODO remove
        //self.optimizer.backward_step(&grads)?;
        
        // for image in &mut self.images {
        //     println!("Updating image");  // TODO remove
        //     match grads.get(image) {
        //         Some(grad) => {
        //             *image = (&*image - grad.affine(learning_rate, 0.0))?.requires_grad()?;
        //         },
        //         None => return Err(anyhow::anyhow!("Gradient not found for image")),
        //     }
        // }
        
        Ok(loss.to_vec0()?)
    }
}

fn main() -> Result<()> {
    let device = Device::cuda_if_available(0)?;
    let mut opt_images = OptimizableImages::new(4, &device)?;
    
    let num_steps = 1000;

    println!("Starting optimization");  // TODO remove
    
    for step in 0..num_steps {
        let loss = opt_images.optimization_step()?;
        println!("vars {:?}", opt_images.varmap.all_vars());  // TODO remove
        // first image values
        // println!("image0: {:?}", opt_images.varmap.get(IMAGE_DIMS, "image0", candle_nn::Init::Randn { mean: 0.0, stdev: 1.0 }, DType::F32, &device)?.to_vec3::<f32>());  // TODO remove
        println!("Step {}: Loss = {}", step, loss);
        if step % 10 == 0 {
            println!("Step {}: Loss = {}", step, loss);
            // TODO remove
            // flatten the tensor to 1d and save
            save_tensor_to_image(&opt_images.varmap.all_vars()[0].as_tensor(), &format!("img{}-step{}.jpg", 0, step))?;
        }
    }
    Ok(())
}
