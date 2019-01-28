##Main run code from:
# Copyright (c) 2015-2018 Anish Athalye. Released under GPLv3.
#Modified from: Anish Athalye (2015). Neural Style [online].
#Accessed December-January 2018. Available from [url https://github.com/anishathalye/neural-style]
#
#Minor changes made to key algorithms
#Check for CLI if not then create GUI
#GUI takes custom user inputs for almost
#every initiated object attribute
#default vars given by Anish Athalye read
#in from separate file


#import python libraries/modules
import os
import wx
import sys
import math
import numpy as np
import scipy.misc as spm
from argparse import ArgumentParser
from collections import OrderedDict
from PIL import Image
from wx.lib.masked import NumCtrl

##import own files
from stylize import stylize
import default_args

# writes command line arguments
# build_parser() func modified slightly
# from original, some variables removed
def build_parser():
    parser = ArgumentParser()
    """
    parser.add_argument('--input', '-F',
                        help='name of input file')
    """
    parser.add_argument('--content',
            dest='content', help='content image',
            metavar='CONTENT', required=False) #set to False to allow for CLI check
    parser.add_argument('--styles',
            dest='styles', help='style image',
            metavar='STYLE', required=False) # set to False to allow for CLI check
    parser.add_argument('--output',
            dest='output', help='output path',
            metavar='OUTPUT', required=False) # set to False to allow for CLI check
    parser.add_argument('--iterations', type=int,
            dest='iterations', help='iterations (default %(default)s)',
            metavar='ITERATIONS', default=default_args.ITERATIONS)
    parser.add_argument('--print-iterations', type=int,
            dest='print_iterations', help='statistics printing frequency',
            metavar='PRINT_ITERATIONS')
    parser.add_argument('--checkpoint-output',
            dest='checkpoint_output', help='checkpoint output format, e.g. output%%s.jpg',
            metavar='OUTPUT')
    parser.add_argument('--checkpoint-iterations', type=int,
            dest='checkpoint_iterations', help='checkpoint frequency',
            metavar='CHECKPOINT_ITERATIONS')
    parser.add_argument('--progress-write', default=False, action='store_true',
            help="write iteration progess data to OUTPUT's dir",
            required=False)
    parser.add_argument('--progress-plot', default=False, action='store_true',
            help="plot iteration progess data to OUTPUT's dir",
            required=False)
    parser.add_argument('--width', type=int,
            dest='width', help='output width',
            metavar='WIDTH')
    parser.add_argument('--style-scale', type=float,
            dest='style_scale',
            nargs='+', help='style scale',
            metavar='STYLE_SCALE')
    parser.add_argument('--network',
            dest='network', help='path to network parameters (default %(default)s)',
            metavar='VGG_PATH', default=default_args.VGG_PATH)
    parser.add_argument('--content-weight-blend', type=float,
            dest='content_weight_blend', help='content weight blend, conv4_2 * blend + conv5_2 * (1-blend) (default %(default)s)',
            metavar='CONTENT_WEIGHT_BLEND', default=default_args.CONTENT_WEIGHT_BLEND)
    parser.add_argument('--content-weight', type=float,
            dest='content_weight', help='content weight (default %(default)s)',
            metavar='CONTENT_WEIGHT', default=default_args.CONTENT_WEIGHT)
    parser.add_argument('--style-weight', type=float,
            dest='style_weight', help='style weight (default %(default)s)',
            metavar='STYLE_WEIGHT', default=default_args.STYLE_WEIGHT)
    parser.add_argument('--style-layer-weight-exp', type=float,
            dest='style_layer_weight_exp', help='style layer weight exponentional increase - weight(layer<n+1>) = weight_exp*weight(layer<n>) (default %(default)s)',
            metavar='STYLE_LAYER_WEIGHT_EXP', default=default_args.STYLE_LAYER_WEIGHT_EXP)
    parser.add_argument('--tv-weight', type=float,
            dest='tv_weight', help='total variation regularization weight (default %(default)s)',
            metavar='TV_WEIGHT', default=default_args.TV_WEIGHT)
    parser.add_argument('--learning-rate', type=float,
            dest='learning_rate', help='learning rate (default %(default)s)',
            metavar='LEARNING_RATE', default=default_args.LEARNING_RATE)
    parser.add_argument('--beta1', type=float,
            dest='beta1', help='Adam: beta1 parameter (default %(default)s)',
            metavar='BETA1', default=default_args.BETA1)
    parser.add_argument('--beta2', type=float,
            dest='beta2', help='Adam: beta2 parameter (default %(default)s)',
            metavar='BETA2', default=default_args.BETA2)
    parser.add_argument('--eps', type=float,
            dest='epsilon', help='Adam: epsilon parameter (default %(default)s)',
            metavar='EPSILON', default=default_args.EPSILON)
    parser.add_argument('--initial',
            dest='initial', help='initial image',
            metavar='INITIAL')
    parser.add_argument('--initial-noiseblend', type=float,
            dest='initial_noiseblend', help='ratio of blending initial image with normalized noise (if no initial image specified, content image is used) (default %(default)s)',
            metavar='INITIAL_NOISEBLEND')
    parser.add_argument('--preserve-colors', action='store_true',
            dest='preserve_colors', help='style-only transfer (preserving colors) - if color transfer is not needed')
    parser.add_argument('--pooling',
            dest='pooling', help='pooling layer configuration: max or avg (default %(default)s)',
            metavar='POOLING', default=default_args.POOLING)
    parser.add_argument('--overwrite', action='store_true',
            dest='overwrite', help='write file even if there is already a file with that name')
    return parser
##---end Anish Athalye parser code



class OptionsG(object):

    def __init__(self, styles, style_weight, style_layer_weight_exp, network, content, content_weight, \
                    content_weight_blend, output, iterations, width, style_scale,  \
                    initial, initial_noiseblend, checkpoint_output, preserve_colors, overwrite, \
                    tv_weight, learning_rate, beta1, beta2, epsilon, pooling, print_iterations, \
                    checkpoint_iterations, progress_plot, progress_write):
        self.styles = styles
        self.style_weight = style_weight
        self.style_layer_weight_exp = style_layer_weight_exp
        self.network = network
        self.content = content
        self.content_weight = content_weight
        self.content_weight_blend = content_weight_blend
        self.output = output
        self.iterations = iterations
        self.width = width
        self.style_scale = style_scale
        self.initial = initial
        self.initial_noiseblend = initial_noiseblend
        self.checkpoint_output = checkpoint_output
        self.preserve_colors = preserve_colors
        self.overwrite = overwrite
        self.tv_weight = tv_weight
        self.learning_rate = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.pooling = pooling
        self.print_iterations = print_iterations
        self.checkpoint_iterations = checkpoint_iterations
        self.progress_plot = progress_plot
        self.progress_write = progress_write

wildcard = "Image source (*.jpg)|*.jpg|" \
            "All files (*.*)|*.*"

## GUI functions referenced from:
## http://zetcode.com/wxpython/


class CreateGui(wx.Frame):

    def __init__(self, parent, id):
        wx.Frame.__init__(self,parent,id,'GUI')
        self.initUI()
        self.currentDirectory = os.getcwd()

    def initUI(self):

        ##init options to default to put data into
        options = OptionsG(None, default_args.STYLE_WEIGHT, default_args.STYLE_LAYER_WEIGHT_EXP, default_args.VGG_PATH, \
                        None, default_args.CONTENT_WEIGHT, default_args.CONTENT_WEIGHT_BLEND, None, default_args.ITERATIONS, \
                        None, default_args.STYLE_SCALE, None, None, 'cpop%s.jpg', False, False, \
                        default_args.TV_WEIGHT, default_args.LEARNING_RATE, default_args.BETA1, \
                        default_args.BETA2, default_args.EPSILON, default_args.POOLING, 100, 100, True, True)

        ##init gui panel
        panel = wx.Panel(self)
        guiBox = wx.BoxSizer(wx.VERTICAL)

        required = wx.StaticBox(panel, label = 'Required')
        requireds = wx.StaticBoxSizer(required, orient = wx.HORIZONTAL)
        #required variables
        contentB = wx.Button(panel, label = 'Content Image')
        contentB.Bind(wx.EVT_BUTTON, self.onOpenContentFile)
        styleB = wx.Button(panel, label = 'Style Image')
        styleB.Bind(wx.EVT_BUTTON, self.onOpenStyleFile)
        output = wx.Button(panel, label = 'Output')
        output.Bind(wx.EVT_BUTTON, self.onSaveFile)

        requireds.Add(contentB, 0, wx.ALL, 5)
        requireds.Add(styleB, 0, wx.ALL, 5)
        requireds.Add(output, 0, wx.ALL, 5)

        #optional settings (if not default)
        optional = wx.StaticBox(panel, label = 'Optional')
        optionals = wx.StaticBoxSizer(optional, orient = wx.HORIZONTAL)

#col1 = wx.StaticBox(panel)
        col1s = wx.BoxSizer( wx.VERTICAL)

        _style_weightBox = wx.BoxSizer(wx.HORIZONTAL)
        _style_weightLabel = wx.StaticText(panel, label = 'style_weight')
        self._style_weight = wx.SpinCtrl(panel, value = str(options.style_weight), min=0, max=1000)
        _style_weightBox.Add(_style_weightLabel)
        _style_weightBox.Add(self._style_weight)
        #_style_weight.SetRange(0,1000)

        _style_layer_weight_expBox = wx.BoxSizer(wx.HORIZONTAL)
        _style_layer_weight_expLabel = wx.StaticText(panel, label = 'style_layer_weight_exp')
        self._style_layer_weight_exp = wx.SpinCtrl(panel, value = str(options.style_layer_weight_exp), min=0.0, max=2.0)
        _style_layer_weight_expBox.Add(_style_layer_weight_expLabel)
        _style_layer_weight_expBox.Add(self._style_layer_weight_exp)
        #_style_layer_weight_exp.SetRange(0.0,2.0)

        _networkBox = wx.BoxSizer(wx.HORIZONTAL)
        _networkB = wx.Button (panel, label = 'network')
        _networkB.Bind(wx.EVT_BUTTON, self.onOpenNetworkFile)
        _networkBox.Add(_networkB, 0, wx.ALL, 5)

        _content_weightBox = wx.BoxSizer(wx.HORIZONTAL)
        _content_weightLabel = wx.StaticText(panel, label = 'content_weight')
        self._content_weight = wx.SpinCtrl(panel, value = str(options.content_weight), min=0, max=10000)
        _content_weightBox.Add(_content_weightLabel)
        _content_weightBox.Add(self._content_weight)
        #_content_weight.SetRange(0,10000)      

        _content_weight_blendBox = wx.BoxSizer(wx.HORIZONTAL)
        _content_weight_blendLabel = wx.StaticText(panel, label = 'content_weight_blend')
        self._content_weight_blend = wx.SpinCtrl(panel, value = str(options.content_weight_blend), min=0.0, max=1.0)
        _content_weight_blendBox.Add(_content_weight_blendLabel)
        _content_weight_blendBox.Add(self._content_weight_blend)
        #_content_weight_blend.SetRange(0.0,1.0)

        _iterationsBox = wx.BoxSizer(wx.HORIZONTAL)
        _iterationsLabel = wx.StaticText(panel, label = 'iterations')
        self._iterations = wx.SpinCtrl(panel, value = str(options.iterations), min=0, max=2000)
        _iterationsBox.Add(_iterationsLabel)
        _iterationsBox.Add(self._iterations)
        #_iterations.SetRange(0,2000)   #allow for 0/1 to check running

        col1s.Add(_style_weightBox)
        col1s.Add(_style_layer_weight_expBox)
        col1s.Add(_networkBox)
        col1s.Add(_content_weightBox)
        col1s.Add(_content_weight_blendBox)
        col1s.Add(_iterationsBox)

#  col2 = wx.StaticBox(panel)
        col2s = wx.BoxSizer(  wx.VERTICAL)

        _widthBox = wx.BoxSizer(wx.HORIZONTAL)
        _widthLabel = wx.StaticText(panel, label = 'width (pixels)')
        self._width = NumCtrl(panel)
        _widthBox.Add(_widthLabel)
        _widthBox.Add(self._width)

        _style_scaleBox = wx.BoxSizer(wx.HORIZONTAL)
        _style_scaleLabel = wx.StaticText(panel, label = 'style_scale')
        self._style_scale = wx.SpinCtrl(panel, value = str(options.style_scale), min=0.0, max=1.0)
        _style_scaleBox.Add(_style_scaleLabel)
        _style_scaleBox.Add(self._style_scale)
        #_style_scale.SetRange(0.0,1.0)    

        _initial_noiseblendBox = wx.BoxSizer(wx.HORIZONTAL)
        _initial_noiseblendLabel = wx.StaticText(panel, label = 'initial_noiseblend')
        self._initial_noiseblend = wx.SpinCtrl(panel, value = str(options.initial_noiseblend), min=0.0, max=1.0)
        _initial_noiseblendBox.Add(_initial_noiseblendLabel)
        _initial_noiseblendBox.Add(self._initial_noiseblend)
        #_initial_noiseblend.SetRange(0.0,1.0)   

        _overwriteBox = wx.BoxSizer(wx.HORIZONTAL)
        self._overwriteCB = wx.CheckBox(panel, label = 'overwrite')
        self._overwriteCB.SetValue(options.overwrite)
        self._overwriteCB.Bind(wx.EVT_CHECKBOX, self.overwriteToggle)
        _overwriteBox.Add(self._overwriteCB)

        _preserve_colorsBox = wx.BoxSizer(wx.HORIZONTAL)
        self._preserve_colorsCB = wx.CheckBox(panel, label = 'preserve_colors')
        self._preserve_colorsCB.SetValue(options.preserve_colors)
        self._preserve_colorsCB.Bind(wx.EVT_CHECKBOX, self.presColToggle)
        _preserve_colorsBox.Add(self._preserve_colorsCB)

        _tv_weightBox = wx.BoxSizer(wx.HORIZONTAL)
        _tv_weightLabel = wx.StaticText(panel, label = 'tv_weight')
        self._tv_weight = wx.SpinCtrl(panel, value = str(options.tv_weight), min=0, max=100)
        _tv_weightBox.Add(_tv_weightLabel)
        _tv_weightBox.Add(self._tv_weight)
        #_tv_weight.SetRange(0,100)    

        col2s.Add(_widthBox)
        col2s.Add(_style_scaleBox)
        col2s.Add(_initial_noiseblendBox)
        col2s.Add(_overwriteBox)
        col2s.Add(_preserve_colorsBox)
        col2s.Add(_tv_weightBox)

#    col3 = wx.StaticBox(panel)
        col3s = wx.BoxSizer( wx.VERTICAL)
     

        _learning_rateBox = wx.BoxSizer(wx.HORIZONTAL)
        _learning_rateLabel = wx.StaticText(panel, label = 'learning_rate')
        self._learning_rate = wx.SpinCtrl(panel, value = str(options.learning_rate), min=0, max=100)
        _learning_rateBox.Add(_learning_rateLabel)
        _learning_rateBox.Add(self._learning_rate)        
        #_learning_rate.SetRange(0,100)   

        _beta1Box = wx.BoxSizer(wx.HORIZONTAL)
        _beta1Label = wx.StaticText(panel, label = 'beta1')
        self._beta1 = wx.SpinCtrl(panel, value = str(options.beta1), min=0.000, max=1.000)
        _beta1Box.Add(_beta1Label)
        _beta1Box.Add(self._beta1)
        #_beta1.SetRange(0.000,1.000)

        _beta2Box = wx.BoxSizer(wx.HORIZONTAL)
        _beta2Label = wx.StaticText(panel, label = 'beta2')
        self._beta2 = wx.SpinCtrl(panel, value = str(options.beta2), min=0.000, max=1.000)
        _beta2Box.Add(_beta2Label)
        _beta2Box.Add(self._beta2)
        #_beta2.SetRange(0.000,1.000)

        _epsilonBox = wx.BoxSizer(wx.HORIZONTAL)
        _epsilonLabel = wx.StaticText(panel, label = 'epsilon')
        self._epsilon = NumCtrl(panel, value = np.format_float_scientific(options.epsilon))
        _epsilonBox.Add(_epsilonLabel)
        _epsilonBox.Add(self._epsilon)

        poolOpts = ['max','avg']
        _poolingBox = wx.BoxSizer(wx.HORIZONTAL)
        _poolingLabel = wx.StaticText(panel, label = 'pooling')
        self._pooling = wx.ComboBox(panel, choices=poolOpts, style = wx.CB_READONLY)
        self._pooling.Bind(wx.EVT_COMBOBOX, self.onPoolSelect)
        _poolingBox.Add(_poolingLabel)
        _poolingBox.Add(self._pooling)

        _progress_plotBox = wx.BoxSizer(wx.HORIZONTAL)
        self._progress_plotCB = wx.CheckBox(panel, label = 'progress_plot')
        self._progress_plotCB.SetValue(options.progress_plot)
        _progress_plotBox.Add(self._progress_plotCB)

        _progress_writeBox = wx.BoxSizer(wx.HORIZONTAL)
        self._progress_writeCB = wx.CheckBox(panel, label = 'progress_write')
        self._progress_writeCB.SetValue(options.progress_write)
        _progress_writeBox.Add(self._progress_writeCB)

        col3s.Add(_learning_rateBox)
        col3s.Add(_beta1Box)
        col3s.Add(_beta2Box)
        col3s.Add(_epsilonBox)
        col3s.Add(_poolingBox)
        col3s.Add(_progress_plotBox)
        col3s.Add(_progress_writeBox)

        optionals.Add(col1s)
        optionals.Add(col2s)
        optionals.Add(col3s)

#buttons + gui collect
        guiBox.Add(requireds, flag = wx.EXPAND)
        guiBox.Add(optionals)

        buttonsbox=wx.BoxSizer(wx.HORIZONTAL)
        button = wx.Button(panel,label="Go", size = (40,40))
        button.Bind(wx.EVT_BUTTON, self.goButton)
        exitbutton=wx.Button(panel, label = "Exit", size = (40, 40))
        exitbutton.Bind(wx.EVT_BUTTON, self.onExit)

        buttonsbox.Add(button)
        buttonsbox.Add(exitbutton)

        guiBox.Add(buttonsbox, border = 5)

        guiBox.SetSizeHints(self)

        guiBox.Add(panel, proportion = 1, flag = wx.ALIGN_LEFT|wx.ALIGN_TOP|wx.ALL, border = 5)

        self.SetSizer(guiBox)

##http://www.blog.pythonlibrary.org/2010/06/26/the-dialogs-of-wxpython-part-1-of-2/ ##open/output
    def onOpenContentFile(self, event):
        """
        Create and show the Open FileDialog
        """
        dlg = wx.FileDialog(
            self, message="Choose a file",
            defaultDir=self.currentDirectory, 
            defaultFile=".jpg",
            wildcard=wildcard,
            style=wx.FD_OPEN | wx.FD_MULTIPLE | wx.FD_CHANGE_DIR
            )
        if dlg.ShowModal() == wx.ID_OK:
            print "opened content"
            contentimg = dlg.GetFilename()
            options.content = sys.path[0] + "/examples/" + contentimg
            options.width = Image.open(contentimg).size[0]
            options.initial = options.content
            print options.content
        dlg.Destroy()

    def onOpenStyleFile(self, event):
        """
        Create and show the Open FileDialog
        """
        dlg = wx.FileDialog(
            self, message="Choose a file",
            defaultDir=self.currentDirectory, 
            defaultFile=".jpg",
            wildcard=wildcard,
            style=wx.FD_OPEN | wx.FD_MULTIPLE | wx.FD_CHANGE_DIR
            )
        if dlg.ShowModal() == wx.ID_OK:
            print "opened style"
            styleimg = dlg.GetFilename()
            options.styles = sys.path[0] + "/examples/" + styleimg
            print options.styles
        dlg.Destroy()

    def onOpenNetworkFile(self, event):
        """
        Create and show the Open FileDialog
        """
        dlg = wx.FileDialog(
            self, message="Choose a file",
            defaultDir=self.currentDirectory, 
            defaultFile=".mat",
            wildcard=wildcard,
            style=wx.FD_OPEN | wx.FD_MULTIPLE | wx.FD_CHANGE_DIR
            )
        if dlg.ShowModal() == wx.ID_OK:
            print "opened network"
            _network = dlg.GetFilename()
            options.network = sys.path[0] + '/' + _network
            print options.network
        dlg.Destroy()

    def onSaveFile(self, event):
        """
        Create and show the Save FileDialog
        """
        dlg = wx.FileDialog(
            self, message="Save file as ...", 
            defaultDir=self.currentDirectory, 
            defaultFile=".jpg", wildcard=wildcard, style=wx.FD_SAVE
            )
        if dlg.ShowModal() == wx.ID_OK:
            print "saved"
            outputimg = dlg.GetFilename()
            options.output = sys.path[0] + "/" + outputimg
            print options.output

        dlg.Destroy()
    
    def overwriteToggle(self, event):
        overwriteT = event.GetEventObject()
        isChecked = overwriteT.GetValue()

        if isChecked:
            self._overwriteCB = True
        else:
            self._overwriteCB = False

    def presColToggle(self, event):
        presColT = event.GetEventObject()
        isChecked = presColT.GetValue()

        if isChecked:
            self._preserve_colorsCB = True
        else:
            self._preserve_colorsCB = False


    def onPoolSelect(self, event):
        pool = event.GetString()
        self._pooling.SetValue(pool)


##----------------------------------------

    def goButton(self, event):
        #---------extra test set vars------------
                ##set options to be self
                ##options.styles = self.styles
                ##options.content = self.content
                ##options.output = self.output
                ##options._network = self.network.GetValue() #set in Network func
                ##options.initial = self._initial.GetValue() #initial = content
        options.style_weight = self._style_weight.GetValue()
        options.style_layer_weight_exp = self._style_layer_weight_exp.GetValue()
        options.content_weight = self._content_weight.GetValue()
        options.content_weight_blend = self._content_weight_blend.GetValue()
        options.iterations = self._iterations.GetValue()
        options.style_scale = self._style_scale.GetValue()
        options.initial_noiseblend = self._initial_noiseblend.GetValue()
        options.preserve_colors = self._preserve_colorsCB.GetValue()
        options.overwrite = self._overwriteCB.GetValue()
        options.tv_weight = self._tv_weight.GetValue()
        options.learning_rate = self._learning_rate.GetValue()
        options.beta1 = self._beta1.GetValue()
        options.beta2 = self._beta2.GetValue()
        options.epsilon = self._epsilon.GetValue()
        options.pooling = self._pooling.GetValue()
        options.progress_plot = self._progress_plotCB.GetValue()
        options.progress_write = self._progress_writeCB.GetValue()
        #----------------------------------------
        #---set individually due to list setting causing segfault--
        #----------------------------------------
        #print options.overwrite #options.overwrite = True? Getting overriden in run()?
        #print os.getcwd() ##for debugging/checking path manually       
       
        os.chdir(sys.path[0])
        print "Please wait..."

        run()
        
    def onExit(self, event):

        self.Close(True)



def make_gui(options):

        options = options
        app = wx.App()
        frame = CreateGui(parent = None, id=-1)
        frame.Show()
        app.MainLoop()


## Begin Anish Athalye citation
## small changes made due to segmentation
## faults arising from vars named the same
## as libs/lib functions
## style blend also removed

def run():
    # https://stackoverflow.com/a/42121886
    key = 'TF_CPP_MIN_LOG_LEVEL'
    if key not in os.environ:
        os.environ[key] = '2'


    if not os.path.isfile(options.network):
        print os.getcwd()
        parser.error("Network %s does not exist. (Did you forget to download it?)" % options.network)

    content_image = imageread(options.content)
    style_image = imageread(options.styles)


    width = options.width
    if width is not None:
        new_shape = (int(math.floor(float(content_image.shape[0]) /
                content_image.shape[1] * width)), width)
        content_image = spm.imresize(content_image, new_shape)
    target_shape = content_image.shape
    style_scale = default_args.STYLE_SCALE
    style_image = spm.imresize(style_image, style_scale *
            target_shape[1] / style_image.shape[1])


    initial = options.initial
    if initial is not None:
        initial = spm.imresize(imageread(initial), content_image.shape[:2])
        # Initial guess is specified, but not noiseblend - no noise should be blended
        if options.initial_noiseblend is None:
            options.initial_noiseblend = 0.0
    else:
        # Neither inital, nor noiseblend is provided, falling back to random generated initial guess
        if options.initial_noiseblend is None:
            options.initial_noiseblend = 1.0
        if options.initial_noiseblend < 1.0:
            initial = content_image

    if options.checkpoint_output and "%s" not in options.checkpoint_output:
        parser.error("To save intermediate images, the checkpoint output "
                     "parameter must contain `%s` (e.g. `foo%s.jpg`)")


    #print options.overwrite ##overwrite = False? is getting overriden after run() called - unsure why
    # try saving a dummy image to the output path to make sure that it's writable
    if os.path.isfile(options.output) and not options.overwrite:
        raise IOError("%s already exists, will not replace it without the '--overwrite' flag" % options.output)
    try:
        imsave(options.output, np.zeros((500, 500, 3)))
    except:
        raise IOError('%s is not writable or does not have a valid file extension for an image file' % options.output)

    loss_arrs = None
    for iteration, image, loss_vals in stylize(
        network=options.network,
        initial=initial,
        initial_noiseblend=options.initial_noiseblend,
        content=content_image,
        styles=style_image,
        preserve_colors=options.preserve_colors,
        iterations=options.iterations,
        content_weight=options.content_weight,
        content_weight_blend=options.content_weight_blend,
        style_weight=options.style_weight,
        style_layer_weight_exp=options.style_layer_weight_exp,
        tv_weight=options.tv_weight,
        learning_rate=options.learning_rate,
        beta1=options.beta1,
        beta2=options.beta2,
        epsilon=options.epsilon,
        pooling=options.pooling,
        print_iterations=options.print_iterations,
        checkpoint_iterations=options.checkpoint_iterations,
    ):
        if (image is not None) and options.checkpoint_output:
            imsave(options.checkpoint_output % iteration, image)
        if (loss_vals is not None) \
                and (options.progress_plot or options.progress_write):
            if loss_arrs is None:
                itr = []
                loss_arrs = OrderedDict((key, []) for key in loss_vals.keys())
            for key,val in loss_vals.items():
                loss_arrs[key].append(val)
            itr.append(iteration)

    print options.output
    imsave(options.output, image)

    if options.progress_write:
        fn = "{}/progress.txt".format(os.path.dirname(options.output))
        tmp = np.empty((len(itr), len(loss_arrs)+1), dtype=float)
        tmp[:,0] = np.array(itr)
        for ii,val in enumerate(loss_arrs.values()):
            tmp[:,ii+1] = np.array(val)
        np.savetxt(fn, tmp, header=' '.join(['itr'] + list(loss_arrs.keys())))


    if options.progress_plot:
        import matplotlib
        matplotlib.use('Agg')
        from matplotlib import pyplot as plt
        fig,ax = plt.subplots()
        for key, val in loss_arrs.items():
            ax.semilogy(itr, val, label=key)
        ax.legend()
        ax.set_xlabel("iterations")
        ax.set_ylabel("loss")
        fig.savefig("{}/progress.png".format(os.path.dirname(options.output)))


def imageread(path):
#    print "path " + path    #debug to check calling correctly
    img = spm.imread(path).astype(np.float)
    if len(img.shape) == 2:
        # greyscale
        img = np.dstack((img,img,img))
    elif img.shape[2] == 4:
        # PNG with alpha channel
        img = img[:,:,:3]
    return img


def imsave(path, img):
    img = np.clip(img, 0, 255).astype(np.uint8)
    Image.fromarray(img).save(path, quality=95)

## ----- End Anish Athalye citation


# CLI/GUI branch referenced from:
# Caird, A., 2019. A Simple GUI and Command-line Python 
# Program with a file browser! and an exit button!! zomg!!! [online].
# Sitewide ATOM. Available from: https://acaird.github.io/2016/02/07/simple-python-gui

if __name__ == '__main__':

    parser = build_parser()
    options = parser.parse_args()

    if (options.content):              # If there is a CLI arg,
        options
        run()     # run the command-line version
    else:
        make_gui(options)
        
                   # otherwise run the GUI version

