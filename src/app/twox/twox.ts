import {ImagePlane} from './ImagePlane';
import {ChannelImage} from './ChannelImage';
import {Array1D, Array2D, NDArrayMathGPU, Scalar} from 'deeplearn';
import {conv_util} from 'deeplearn';
import {HttpClient, HttpClientJsonpModule, HttpClientModule} from '@angular/common/http';
import {TwoxComponent} from './twox.component';

export class Twox {
  scale2xModel;
  noiseModel;
  scale;
  isDenoising;
  URL;

  constructor(json, private http: HttpClient) {
    this.scale2xModel = json.scale2xModel;
    this.noiseModel = json.noiseModel;
    this.scale = json.scale;
    this.isDenoising = json.isDenoising;
  }

  static normalize(image) {
    let width = image.width;
    let height = image.height;
    let imagePlane = new ImagePlane(width, height, null);
    if (imagePlane.getBuffer().length != image.buffer.length) {
      throw new Error('Assertion error: length');
    }
    for (let i = 0; i < image.buffer.length; i++) {
      imagePlane.setValueIndexed(i, image.buffer[i] / 255.0);
    }
    return imagePlane;
  }

  static denormalize(imagePlane) {
    let image = new ChannelImage(imagePlane.width, imagePlane.height, null);

    for (let i = 0; i < imagePlane.getBuffer().length; i++) {
      image.buffer[i] = Math.round(imagePlane.getValueIndexed(i) * 255.0);
    }
    return image;
  }

  static convolution(inputPlanes, W, nOutputPlane, bias) {
    let width = inputPlanes[0].width;
    let height = inputPlanes[0].height;
    let outputPlanes = [];
    for (let o = 0; o < nOutputPlane; o++) {
      outputPlanes[o] = new ImagePlane(width - 2, height - 2, null);
    }
    let sumValues = new Float32Array(nOutputPlane);
    let biasValues = new Float32Array(nOutputPlane);
    biasValues.set(bias);
    for (let w = 1; w < width - 1; w++) {
      for (let h = 1; h < height - 1; h++) {
        sumValues.set(biasValues);  // leaky ReLU bias
        for (let i = 0; i < inputPlanes.length; i++) {
          let i00 = inputPlanes[i].getValue(w - 1, h - 1);
          let i10 = inputPlanes[i].getValue(w, h - 1);
          let i20 = inputPlanes[i].getValue(w + 1, h - 1);
          let i01 = inputPlanes[i].getValue(w - 1, h);
          let i11 = inputPlanes[i].getValue(w, h);
          let i21 = inputPlanes[i].getValue(w + 1, h);
          let i02 = inputPlanes[i].getValue(w - 1, h + 1);
          let i12 = inputPlanes[i].getValue(w, h + 1);
          let i22 = inputPlanes[i].getValue(w + 1, h + 1);

          for (let o = 0; o < nOutputPlane; o++) {
            // assert inputPlanes.length == params.weight[o].length
            let weight_index = (o * inputPlanes.length * 9) + (i * 9);
            let value = sumValues[o];
            value += i00 * W[weight_index++];
            value += i10 * W[weight_index++];
            value += i20 * W[weight_index++];
            value += i01 * W[weight_index++];
            value += i11 * W[weight_index++];
            value += i21 * W[weight_index++];
            value += i02 * W[weight_index++];
            value += i12 * W[weight_index++];
            value += i22 * W[weight_index++];
            sumValues[o] = value;
          }
        }
        for (let o = 0; o < nOutputPlane; o++) {
          let v = sumValues[o];
          // v += bias[o]; // leaky ReLU bias is already added above
          if (v < 0) {
            v *= 0.1;
          }
          outputPlanes[o].setValue(w - 1, h - 1, v);
        }
      }
    }

    //TODO
    // for (let o = 0; o < inputPlanes.length; o++) {
    //   outputPlanes[o] = new ImagePlane(width - 2, height - 2, null);
    //   const math = new NDArrayMathGPU();
    //   console.log(inputPlanes[o].toString());
    //   let inPlane = Array2D.new([3, 3], inputPlanes[o].valueOf(), 'float32');
    //   let convInfo = math.conv2d(inputPlanes[o], W[o], bias[o], 1, 'valid');
    //   console.log(convInfo.outShape.toString());
    // }

    console.log('Convolution successful.');
    return outputPlanes;
  }

  static typeW(model) {
    console.log('Initialize typed W matrix');
    let W = [];
    for (let l = 0; l < model.length; l++) {
      // initialize weight matrix
      let layerWeight = model[l].weight;
      let arrayW = [];
      layerWeight.forEach(function (weightForOutputPlane) {
        weightForOutputPlane.forEach(function (weightMatrix) {
          weightMatrix.forEach(function (weightVector) {
            weightVector.forEach(function (w) {
              arrayW.push(w);
            });
          });
        });
      });
      let w = new Float32Array(arrayW.length);
      w.set(arrayW);
      W[l] = w;
    }
    return W;
  }

  static calcRGB(imageR, imageG, imageB, model, scale) {
    let [planeR, planeG, planeB] = [imageR, imageG, imageB].map((image) => {
      let imgResized = scale == 1 ? image : image.resize(scale);

      // extrapolation for layer count (each convolution removes outer 1 pixel border)
      let imgExtra = imgResized.extrapolation(model.length);

      return Twox.normalize(imgExtra);
    });

    let inputPlanes = [planeR, planeG, planeB];

    // blocking
    let [inputBlocks, blocksW, blocksH] = ImagePlane.blocking(inputPlanes);
    inputPlanes = null;

    // init W
    let W = Twox.typeW(model);

    let outputBlocks = [];
    for (let b = 0; b < (<any[]>inputBlocks).length; b++) {
      let inputBlock = inputBlocks[b];
      let outputBlock = null;
      for (let l = 0; l < model.length; l++) {
        let nOutputPlane = model[l].nOutputPlane;

        // convolution
        outputBlock = Twox.convolution(inputBlock, W[l], nOutputPlane, model[l]['bias']);
        inputBlock = outputBlock; // propagate output plane to next layer input
        inputBlocks[b] = null;
      }
      outputBlocks[b] = outputBlock;
      let doneRatio = Math.round((100 * (b + 1)) / (<any[]>inputBlocks).length);
      // progress(phase, doneRatio, (<any[]>inputBlocks).length, b + 1);
      console.log('b:' + b + ' is done. ' + Math.round((100 * (b + 1)) / (<any[]>inputBlocks).length) + '%');
    }
    inputBlocks = null;

    // de-blocking
    let outputPlanes = ImagePlane.deblocking(outputBlocks, blocksW, blocksH);
    if (outputPlanes.length != 3) {
      throw new Error('Output planes must be 3: color channel R, G, B.');
    }

    [imageR, imageG, imageB] = outputPlanes.map((outputPlane) => {
      return Twox.denormalize(outputPlane);
    });

    return [imageR, imageG, imageB];
  }

  calc(image, width, height) {
    let outimage = document.getElementById('outimage')  as HTMLImageElement;
    if (this.scale2xModel == null && this.noiseModel == null) {
      // do nothing
      // done(image, width, height);
      console.log('Nothing to do');
      return;
    }

    // decompose
    console.log('decompose');
    let [imageR, imageG, imageB, imageA] = ChannelImage.channelDecompose(image, width, height);

    // de-noising
    if (this.noiseModel != null) {
      console.log('Denoising all blocks');
      [imageR, imageG, imageB] = Twox.calcRGB(imageR, imageG, imageB, this.noiseModel, 1);
      console.log('Denoised all blocks');
    }

    // calculate
    if (this.scale2xModel != null) {
      console.log('Scaling all blocks');
      [imageR, imageG, imageB] = Twox.calcRGB(imageR, imageG, imageB, this.scale2xModel, this.scale);
      console.log('Scaled all blocks');
    }

    // resize alpha channel
    imageA = this.scale == 1 ? imageA : imageA.resize(this.scale);
    console.log('recompose');
    let image2x = ChannelImage.channelCompose(imageR, imageG, imageB, imageA);
    // console.log(image2x.toString());
    console.log('Finished.');
    let c = <HTMLCanvasElement>document.getElementById('outCanvas');
    let ctx = c.getContext('2d');
    ctx.canvas.width = width * 2;
    ctx.canvas.height = height * 2;
    let imageData2x = ctx.createImageData(width * 2, height * 2);
    imageData2x.data.set(image2x);
    ctx.putImageData(imageData2x, 0, 0);
  }

  printURL() {
    console.log(this['URL']);
  }

  start() {
    const modelURL =
      'https://raw.githubusercontent.com/takuyaa/waifu2x-js/master/demo/src/www/models/anime_style_art_rgb/scale2.0x_model.json';
    this.http.get(modelURL)
      .subscribe(data => {
        // Read the result field from the JSON response.
        this.scale2xModel = data;
        this.noiseModel = null;
        this.scale = 2;
        this.isDenoising = false;
        let c = <HTMLCanvasElement>document.getElementById('inCanvas');
        let ctx = c.getContext('2d');

        let image = new Image();

        image.onload = () => {
          ctx.canvas.width = image.naturalWidth;
          ctx.canvas.height = image.naturalHeight;
          ctx.clearRect(0, 0, c.width, c.height);
          ctx.drawImage(image, 0, 0, c.width, c.height);
        };
        image.crossOrigin = 'anonymous';
        image.src = this['URL'];

        let superthis = this;
        const loaded = function () {
          console.log('Image loaded.');
          let imageData = ctx.getImageData(0, 0, image.naturalWidth, image.naturalHeight);
          superthis.calc(imageData.data, imageData.width, imageData.height);
        };

        if (image.complete) {
          loaded();
        } else {
          image.addEventListener('load', loaded);
        }
      });
  }
}
