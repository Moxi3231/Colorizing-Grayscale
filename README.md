# Colorizing Grayscale Image
---
## CMPE 258 - Deep Learning
## Dr. Harry Li

---
### Members:
1. Moxank Patel
2. Meet Patel
3. Pratik Shah
4. Swapnil Bagh

----
### Description

Computer Vision presents an intriguing challenge of coloring grayscale images. Researchers have explored various methodologies, such as the Scribble Based method and simple neural networks employing Convolutional Neural Networks (CNN) to accomplish this task. One of the neural network-based approaches is Generative Adversarial Networks (GAN), which involve a min-max game between the generator and the discriminator to minimize the error. This project focuses on using GAN to colorize grayscale images. To enhance the training, we incorporated an Attention Layer that concentrates on features and directly reduces the generator's error. Furthermore, we included batch normalization and residuals in the network to address the vanishing gradient issue.

---


### Requirements

1. tensorflow-gpu == 2.8
2. cv2
3. numpy
4. pandas
5. sklearn
---

### Running the project

Once all the requirements are satisfied run the cell of the notebook.


### Training the model
> wpath = './path/to/saved/weights'

> learning_rate_fn_1 = tf.keras.optimizers.schedules.PolynomialDecay(1e-03,128*8*100*10000, 1e-012, power=0.95)

>learning_rate_fn_2 = tf.keras.optimizers.schedules.PolynomialDecay(1e-05,128*9*100*10000, 1e-013, power=0.95)


> G_opt = tf.optimizers.Adam(learning_rate=learning_rate_fn_1,beta_1=0.5,beta_2=0.95, clipnorm=1.0, epsilon=1e-09,amsgrad=True)

> D_opt = tf.optimizers.Adam(learning_rate=learning_rate_fn_2,beta_1=0.5,beta_2=0.95, clipnorm=1.0, epsilon=1e-09,amsgrad=True)

> gan_model = GAN(img_shape,G_opt,D_opt,weight_path=wpath,load_wei=False)

Note: While creating model for first if weights aren't there then load_wei should be False

### Loading the  model from saved weights
> gan_model = GAN(img_shape,G_opt,D_opt,weight_path=wpath,load_wei=True)


#### Colorizing an image
1. Load the model from saved weights
2. Load the image
3. From the model call the function colorizeImg

> img = cv2.imread('./path/to/image')

> imgs = np.asarray([img])

> output = gan_model.colorizeImg(imgs)

Note: The input image should be normalized in range of [-1,1]. Also image is passed as an array of image

### References

[1]. I. Žeger, S. Grgic, J. Vuković, and G. Šišul, "Grayscale Image Colorization Methods: Overview and Evaluation," in IEEE Access, vol. 9, pp. 113326-113346, 2021, doi: 10.1109/ACCESS.2021.3104515.

[2].  F. Marra, D. Gragnaniello, D. Cozzolino and L. Verdoliva, "Detection of GAN-Generated Fake Images over Social Networks," 2018 IEEE Conference on Multimedia Information Processing and Retrieval (MIPR), Miami, FL, USA, 2018, pp. 384-389, doi: 10.1109/MIPR.2018.00084.