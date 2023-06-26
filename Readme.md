<h1>Start</h1>
<p>1.Download all data. Put train data into /data/train/ folder and test data into /data/test/. Files sample_submission_v2.csv and train_ship_submission_v2.csv into root(project) directory</p>
<p>2.In files train.py, test.py, image_generator.py and data_exploration.ipynb set variable base_directory as a path to base project directory </p>
<p>3.Install all libraries(use requirements.txt)</p>
<p>4.Run train.py. This action will create model directory with pretrained model in it</p>
<p>5.Run test.py to see how model works</p>
<h1>Files description</h1>
<p>1.utils/losses.py - here we can find loss functions that we will use during training</p>
<p>2.utils/image_generator.py - here we can find batch generators for model and predictions that can be shown in matplotlib in test.py file</p>
<p>3.utils/utils.py - here we can find encoders and decoders, data visualization and masks as image file.</p>
<p>3.train.py - here we train our Unet model</p>
<p>4.test.py - here we see result of work of model</p>
<h1>Architecture</h1>
<ul>
<li>
    Architecture: Unet
</li>
<li>
    Loss: FocalLoss
</li>
<li>
    Optimizer: Adam
</li>
<li>
    Metrics: Dice score
</li>
</ul>
<h1>Data exploration</h1>
<p>As we can see from data_exploration.ipynb our starting dataset is very unbalanced because of a big number of images without ship, so I`ve decided to limit number of images with some number of ships to 4000 so model won`t overfit to it</p>