import {MnistData} from './data.js';
var canvas, ctx, saveButton, clearButton, pred;
var pos = {x:0, y:0};
var rawImage;
var model;

function getModel() {
	model = tf.sequential();

	model.add(tf.layers.conv2d({inputShape: [28, 28, 1], kernelSize: 3, filters: 8, activation: 'relu'}));
	model.add(tf.layers.maxPooling2d({poolSize: [2, 2]}));
	model.add(tf.layers.conv2d({filters: 16, kernelSize: 3, activation: 'relu'}));
	model.add(tf.layers.maxPooling2d({poolSize: [2, 2]}));
	model.add(tf.layers.conv2d({filters: 32, kernelSize: 3, activation: 'relu'}));
	model.add(tf.layers.flatten());
	model.add(tf.layers.dense({units: 128, activation: 'relu'}));
	model.add(tf.layers.dense({units: 10, activation: 'softmax'}));

	model.compile({optimizer: tf.train.adam(), loss: 'categoricalCrossentropy', metrics: ['accuracy']});
	return model;
}

async function train(model, data) {
	const metrics = ['loss', 'val_loss', 'acc', 'val_acc'];
	const container = { name: 'Model Training', styles: { height: 640 } };
	const fitCallbacks = tfvis.show.fitCallbacks(container, metrics);

	const BATCH_SIZE = 512;
	const TRAIN_DATA_SIZE = 5500;
	const TEST_DATA_SIZE = 1000;

	const [trainXs, trainYs] = tf.tidy(() => {
		const d = data.nextTrainBatch(TRAIN_DATA_SIZE);
		return [
			d.xs.reshape([TRAIN_DATA_SIZE, 28, 28, 1]), 
			d.labels
		];
	});

	const [testXs, testYs] = tf.tidy(() => {
		const d = data.nextTrainBatch(TEST_DATA_SIZE);
		return [
			d.xs.reshape([TEST_DATA_SIZE, 28, 28, 1]), 
			d.labels
		];
	});

	return model.fit(trainXs, trainYs, {
		batchSize: BATCH_SIZE, 
		validationData: [testXs, testYs], 
		epochs: 20, 
		shuffle: true, 
		callbacks: fitCallbacks
	});
}
function setPosition(e) {
	pos.x = e.clientX-100;
	pos.y = e.clientY-100;
}

function draw(e) {
	console.log("Entering draw " + e.clientX + ", " + e.clientY)
	if (e.buttons != 1) return;
	ctx.beginPath();
	ctx.lineWidth = 24;
	ctx.lineCap = 'round';
	ctx.strokeStyle = 'white';
	ctx.moveTo(pos.x, pos.y);
	setPosition(e);
	ctx.lineTo(pos.x, pos.y);
	ctx.stroke();
	rawImage.src = canvas.toDataURL('image/png');
}

function erase() {
	console.log("Clicked erase button.")
	ctx.fillStyle = "black";
	ctx.fillRect(0, 0, 280, 280);
}

function save() {
	console.log("Clicked classify button.")
	var raw = tf.browser.fromPixels(rawImage, 1);
	var resized = tf.image.resizeBilinear(raw, [28, 28]);
	var tensor = resized.expandDims(0);
	var prediction = model.predict(tensor);
	var pIndex = tf.argMax(prediction, 1).dataSync();
	
	pred.innerHTML = "Prediction: " + pIndex
	alert(pIndex);
}

function init() {
	console.log("Entered init")
	canvas = document.getElementById('canvas');
	rawImage = document.getElementById('canvasimg');
	pred = document.getElementById('pred');

	console.log("Filling black")
	ctx = canvas.getContext("2d");
	erase();

	console.log("Adding EventListener")
	canvas.addEventListener("mousemove", draw);
	canvas.addEventListener("mousedown", setPosition);
	canvas.addEventListener("mouseenter", setPosition);

	console.log("Binding Button")
	saveButton = document.getElementById('sb');
	saveButton.addEventListener("click", save);
	
	clearButton = document.getElementById("cb");
	clearButton.addEventListener("click", erase);
}

async function run() {
	const data = new MnistData();
	await data.load();
	console.log("Mnist Loaded.")
	const model = getModel();
	console.log("Model Loaded.")
	tfvis.show.modelSummary({name: "Model Architecture"}, model);
	await train(model, data);
	console.log("Model Trained.")
	init();
	alert("Finished training. Now you can try to classify your handwriting.")
}

document.addEventListener("DOMContentLoaded", run);