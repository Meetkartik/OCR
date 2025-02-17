const express = require('express');
const multer = require('multer');
const cors = require('cors');
const { PythonShell } = require('python-shell');
// const filePath = require('./ocr_script.py');
const fs = require('fs');

const app = express();
app.use(cors());

const upload = multer({ dest: 'uploads/' });

app.post('/upload', upload.single('file'), async (req, res) => {
    try {
        const filePath = req.file.path;
        const modelType = 'handwritten';

        PythonShell.run('ocr_script.py', { args: [filePath, modelType] }, (err, result) => {
            if (err) {
                console.log(err);
                res.status(500).send
            }
            console.log(result);
        }

        );

    } catch (error) {
        console.log(error);
    }
})

app.listen(5000, () => {
    console.log('Server is running on port 5000');
})