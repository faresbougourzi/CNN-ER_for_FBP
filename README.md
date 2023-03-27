## Deep Learning Based Face Beauty Prediction via Dynamic Robust Losses and Ensemble Regression

In summary, the main contributions of this paper are as follows:

- Inspired by the Att-Unet architecture, we propose three different architectures for segmenting Covid-19 infections from CT-scans. The first variant, Pyramid Att-Unet (PAtt-Unet), uses image pyramids to preserve the spatial awareness in all of the encoder layers. Unlike most attention-based segmentation architectures, our proposed PAtt-Unet uses the attention gates not only in the decoder but also in the encoder.


- Based on PAtt-Unet and DAtt-Unet, we propose a Pyramid Dual-Decoder Att-Unet (PDAtt-Unet) architecture using the pyramid and attention gates to preserve the global spatial awareness in all of the encoder layers. In the decoding phase, PDAtt-Unet has two independent decoders that use the Attention Gates to segment infection and lung simultaneously.

- To address the shortcomings of the binary cross entropy loss function in distinguishing the infection boundaries and the small infection regions, we propose the ${BCE}_{Edge}$ loss that focuses on the edges of the infection regions.

- To evaluate the performance of our proposed architectures, we use four public datasets with two evaluation scenarios (intra and cross datasets),  all slices from CT scans are used for the training and testing phases. 

- To compare the performance of our approach with other CNN-based segmentation architectures, we use three baseline architectures (Unet , Att-Unet and Unet++) and three state-of-the-art architectures for Covid-19 segmentation (Inf-Net, SCOATNet, and nCoVSegNet). The experimental results show the superiority of our proposed architecture compared to the basic segmentation architectures as well as to the three state-of-the-art architectures in both intra-database and inter-database evaluation scenarios. 
