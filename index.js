const express = require('express');
const tf = require('@tensorflow/tfjs-node');
const nsfw = require('nsfwjs');
const fs = require('fs');
const multer = require('multer');
const upload = multer({ dest: 'uploads/' });

const app = express();
const port = 3000;

let _model;

const loadModel = async () => {
  _model = await nsfw.load();
};

// 预先加载模型
loadModel();

app.post('/upload', upload.single('image'), async (req, res) => {
  const image = fs.readFileSync(req.file.path);
  const decodedImage = tf.node.decodeImage(image, 3);
  const predictions = await _model.classify(decodedImage);
  decodedImage.dispose(); // 释放内存
  fs.unlinkSync(req.file.path); // 删除上传的图片

  return res.json(predictions);
});

app.listen(port, () => {
  console.log(`Server started on http://localhost:${port}`);
});
