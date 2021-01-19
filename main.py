from flask import Flask, render_template, request
from flask_uploads import UploadSet, configure_uploads, IMAGES
import flask_crossdomain

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Softmax
from torch.autograd import Variable
from models import FineTuneModel
from config import get_config
import matplotlib.pyplot as plt
import numpy as np
from misc import preprocess_image
import cv2

from datetime import timedelta
from functools import update_wrapper
from flask import make_response, current_app

import base64
from io import BytesIO
from PIL import Image

import datetime

def crossdomain(origin=None, methods=None, headers=None, max_age=21600,
                attach_to_all=True, automatic_options=True):
    """Decorator function that allows crossdomain requests.
      Courtesy of
      https://blog.skyred.fi/articles/better-crossdomain-snippet-for-flask.html
    """
    if methods is not None:
        methods = ', '.join(sorted(x.upper() for x in methods))
    if headers is not None and not isinstance(headers, list):
        headers = ', '.join(x.upper() for x in headers)
    if not isinstance(origin, list):
        origin = ', '.join(origin)
    if isinstance(max_age, timedelta):
        max_age = max_age.total_seconds()

    def get_methods():
        """ Determines which methods are allowed
        """
        if methods is not None:
            return methods

        options_resp = current_app.make_default_options_response()
        return options_resp.headers['allow']

    def decorator(f):
        """The decorator function
        """
        def wrapped_function(*args, **kwargs):
            """Caries out the actual cross domain code
            """
            if automatic_options and request.method == 'OPTIONS':
                resp = current_app.make_default_options_response()
            else:
                resp = make_response(f(*args, **kwargs))
            if not attach_to_all and request.method != 'OPTIONS':
                return resp

            h = resp.headers
            h['Access-Control-Allow-Origin'] = origin
            h['Access-Control-Allow-Methods'] = get_methods()
            h['Access-Control-Max-Age'] = str(max_age)
            h['Access-Control-Allow-Credentials'] = 'true'
            h['Access-Control-Allow-Headers'] = \
                "Origin, X-Requested-With, Content-Type, Accept, Authorization"
            if headers is not None:
                h['Access-Control-Allow-Headers'] = headers
            return resp

        f.provide_automatic_options = False
        return update_wrapper(wrapped_function, f)
    return decorator

# Loading some class and functions
class ModelOutputs():
    """ Class for making a forward pass, and getting:
    1. The network output.
    2. Activations from intermeddiate targetted layers.
    3. Gradients from intermeddiate targetted layers. """
    def __init__(self, model, target_layers, modelname):
        self.model = model
        self.modelname = modelname
        if self.modelname == 'inception_v3':
            self.ex = nn.Sequential(*list(self.model.inceptionV3.children())[:-1])
            self.feature_extractor = FeatureExtractor(self.ex, target_layers, self.modelname)
        elif self.modelname == 'vgg16_bn':
            self.feature_extractor = FeatureExtractor(self.model.features, target_layers, self.modelname)
        elif self.modelname == 'resnet50':
            # self.ex = self.model.features
            self.feature_extractor = FeatureExtractor(self.model.features, target_layers, self.modelname)
        else:
            raise('Not implemented!')

    def get_gradients(self):
        return self.feature_extractor.gradients

    def __call__(self, x):
        target_activations, output  = self.feature_extractor(x)
        if self.modelname == 'inception_v3':
            output = F.avg_pool2d(output, kernel_size=8)
        output = output.view(output.size(0), -1)
        if self.modelname != 'inception_v3':
            output = self.model.classifier(output)
        else:
            output = self.model.inceptionV3.fc(output)
        return target_activations, output

def show_cam_on_image(img, mask, filename, height, width):
    heatmap = cv2.applyColorMap(np.uint8(255*mask), cv2.COLORMAP_JET)
    heatmap = np.float32(heatmap) / 255
    cam = heatmap + np.float32(img)
    cam = cam / np.max(cam)
    cv2.imwrite(filename, np.uint8(255 * cam))
    img = cv2.imread(filename)
    img = cv2.resize(img, (width, height))
    cv2.imwrite(filename, img)

class GradCam:
    def __init__(self, model, target_layer_names, image_size, use_cuda, modelname):
        self.model = model
        self.model.eval()
        self.imsize = image_size
        self.cuda = use_cuda
        if self.cuda:
            self.model = model.cuda()

        self.extractor = ModelOutputs(self.model, target_layer_names, args.arch)

    def forward(self, input):
        return self.model(input) 

    def __call__(self, input, index = None):
        if self.cuda:
            features, output = self.extractor(input.cuda())
        else:
            features, output = self.extractor(input)

        if index == None:
            index = np.argmax(output.cpu().data.numpy())

        one_hot = np.zeros((1, output.size()[-1]), dtype = np.float32)
        one_hot[0][index] = 1
        one_hot = Variable(torch.from_numpy(one_hot), requires_grad = True)
        if self.cuda:
            one_hot = torch.sum(one_hot.cuda() * output)
        else:
            one_hot = torch.sum(one_hot * output)

        self.model.zero_grad()
        one_hot.backward(retain_graph=True)

        grads_val = self.extractor.get_gradients()[-1].cpu().data.numpy()

        target = features[-1]
        target = target.cpu().data.numpy()[0, :]

        weights = np.mean(grads_val, axis = (2, 3))[0, :]
        cam = np.ones(target.shape[1 : ], dtype = np.float32)

        for i, w in enumerate(weights):
            cam += w * target[i, :, :]

        cam = np.maximum(cam, 0)
        cam = cv2.resize(cam, self.imsize)
        cam = cam - np.min(cam)
        cam = cam / np.max(cam)
        return cam, (index + 1)


class FeatureExtractor():
    """ Class for extracting activations and 
    registering gradients from targetted intermediate layers """
    def __init__(self, model, target_layers, modelname):
        self.model = model
        self.modelname = modelname
        self.target_layers = target_layers
        self.gradients = []
        self.afterpool = ['2', '4']  # inception v3 ['Conv2d_2b_3x3', 'Conv2d_4a_3x3']

    def save_gradient(self, grad):
    	self.gradients.append(grad)

    def __call__(self, x):
        outputs = []
        self.gradients = []
        for name, module in self.model._modules.items():
            if self.modelname == 'inception_v3':
                if name != '13': # AuxLogits
                    x = module(x)

                if name in self.afterpool:
                    x = F.max_pool2d(x, kernel_size=3, stride=2)
            else:
                x = module(x)

            if name in self.target_layers:
                x.register_hook(self.save_gradient)
                outputs += [x]

        return outputs, x

# Initialize the flask
app = Flask(__name__, static_folder='uploads')
#app = Flask(__name__, static_folder='.well-known')

photos = UploadSet('photos', IMAGES)

app.config['UPLOADED_PHOTOS_DEST'] = 'uploads/img'
configure_uploads(app, photos)





models = dict()
for cnn, version in [('inception_v3', 'model_0.1.12'), ('vgg16_bn', 'model_0.0.67'), ('resnet50', 'model_0.2.12')]:
    # Initialize the pytorch and model
    args, unparsed = get_config()
    state = {k: v for k, v in args._get_kwargs()}

    args.arch = cnn
    args.checkpoint = 'model/{}/model_best.pth.tar'.format(version)

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id
    use_cuda = torch.cuda.is_available()

    model = FineTuneModel(args.arch, len(args.cls_def))
    print('Loading model from {}'.format(args.checkpoint))

    if use_cuda:
        model.cuda()
        checkpoint = torch.load(args.checkpoint)
    else:
        checkpoint = torch.load(args.checkpoint, map_location=lambda storage, loc: storage)

    model.load_state_dict(checkpoint['state_dict'])
    model.eval()

    print('Model %s is ready!' % args.arch)
    print('Total params: %.2fM' % (sum(p.numel() for p in model.parameters())/1000000.0))

    sm = Softmax(dim=1)

    classeme = ['Normal', 'Inactive Wet', 'Active Wet', 'Drusen']

    # Prepare the heatmap function
    imsize = {'vgg16_bn': (224, 224), 'inception_v3': (299, 299), 'resnet50': (224, 224)}
    targetLookup = {'vgg16_bn': ['42'], 'inception_v3': ['16'], 'resnet50': ['7']}
    grad_cam = GradCam(model=model, target_layer_names=targetLookup[args.arch], image_size=imsize[args.arch], use_cuda=use_cuda, modelname=args.arch)

    models[cnn] = (model, grad_cam, args)


# Set up routing
@app.route("/", methods=['GET', 'POST'])
@crossdomain(origin='*')
def hello():
    upload_filename = None
    heatmap_filename = None
    if request.method == 'POST':
        model, grad_cam, args = models[request.form.get('model')]

        start_time = datetime.datetime.now()

        next_index = len(os.listdir('uploads/img'))

        photo = request.form.get('photo')
        starter = photo.find(',')
        image_data = photo[starter + 1:]
        with open('uploads/img/{}.jpg'.format(next_index), 'wb') as fh:
            fh.write(base64.b64decode(image_data))

        filename = '{}.jpg'.format(next_index)
        #filename = photos.save(request.files['photo'])
        image_path = os.path.join('uploads/img/', filename)

        # Read uploaded image
        img = cv2.imread(image_path, 1)
        height, width, channels = img.shape
        img = np.float32(cv2.resize(img, imsize[args.arch])) / 255
        input = preprocess_image(img, use_cuda)

        # Start classifying
        outputs = model(input)
        outputs = sm(outputs).data
        prob, cls = torch.max(outputs, 1)

        prob = prob.cpu().numpy()[0]
        cls = cls.cpu().numpy()[0]

        upload_filename = filename
        upload_class = classeme[cls]
        upload_prob = prob*100

        end_time = datetime.datetime.now()

        diff = end_time - start_time
        elapsed_ms = (diff.days * 86400000) + (diff.seconds * 1000) + (diff.microseconds / 1000)

        # Start generate heatmap
        # Define the new gnerated filename
        heatmap_filename_0 = os.path.splitext(filename)[0]
        heatmap_filename_1 = os.path.splitext(filename)[1]
        heatmap_filename = heatmap_filename_0 + '-heatmap' + heatmap_filename_1
        heatmap_path = os.path.join('uploads/img/', heatmap_filename)

        target_index = None
        mask, pred = grad_cam(input, target_index)
        show_cam_on_image(img, mask, heatmap_path, height, width)


        # potential risk?
        return render_template('index.html', **locals())
    return render_template('index.html')

