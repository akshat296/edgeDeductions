// https://gammacv.com
const width = 500;
const heigth = 400;
// initialize WebRTC stream and session for runing operations on GPU
const stream = new gm.CaptureVideo(width, heigth);
const sess = new gm.Session();
const canvasProcessed = gm.canvasCreate(width, heigth);

// session uses a context for optimize calculations and prevent recalculations
// context actually a number which help algorythm to run operation efficiently  
let context = 0;
// allocate memeory for storing a frame and calculations output
const input = new gm.Tensor('uint8', [heigth, width, 4]);
// construct operation grap which is actially a Canny Edge Detector
let pipeline = input

pipeline = gm.grayscale(pipeline);
pipeline = gm.gaussianBlur(pipeline, 4, 3);
// pipeline = gm.dilate(pipeline, [1, 1]);
pipeline = gm.sobelOperator(pipeline);
pipeline = gm.cannyEdges(pipeline, 0.25, 0.65);

// initialize graph
sess.init(pipeline);

// allocate output
const output = gm.tensorFrom(pipeline);

// create loop
const tick = () => {
  requestAnimationFrame(tick);
  // Read current in to the tensor
  stream.getImageBuffer(input);

  // finaly run operation on GPU and then write result in to output tensor
  sess.runOp(pipeline, context, output);

  // draw result into canvas
  gm.canvasFromTensor(canvasProcessed, output);

  // if we would like to be graph recalculated we need 
  // to change the context for next frame
  context += 1;
}

function main() {
  // start capturing a camera and run loop
  stream.start();
  tick();

  document.body.children[0].appendChild(canvasProcessed);
}