import numpy as np
import PIL
from io import BytesIO
import matplotlib.pyplot as plt

def plt_to_text(plt_fig=None,width=144,vscale=0.96,colorful=True,invert=False,crop=False,outfile=None):
    """Displays matplotlib figure in terminal as text. You should use a monospcae font like `Cascadia Code PL` to display image correctly. Use before plt.show(). 
    - **Parameters**
        - plt_fig: Matplotlib's figure instance. Auto picks if not given.
        - width  : Character width in terminal, default is 144. Decrease font size when width increased.
        - vscale : Useful to tweek aspect ratio. Default is 0.96 and prints actual aspect in `Cascadia Code PL`. It is approximately `2*width/height` when you select a single space in terminal.
        - colorful: Default is False, prints colored picture if terminal supports it, e.g Windows Terminal.
        - invert  : Defult is False, could be useful for grayscale image. 
        - crop    : Default is False. Crops extra background, can change image color if top left pixel is not in background, in that case set this to False. 
        - outfile: If None, prints to screen. Writes on a file.
    """
    if plt_fig==None:
        plt_fig = plt.gcf()
    plot_bytes = BytesIO()
    plt.savefig(plot_bytes,format='png',dpi=600)
    img = PIL.Image.open(plot_bytes)
    # crop
    if crop:
        bg   = PIL.Image.new(img.mode, img.size, img.getpixel((0,0)))
        diff = PIL.ImageChops.difference(img, bg)
        diff = PIL.ImageChops.add(diff, diff, 2.0, -100) # No idea how it works
        bbox = diff.getbbox()
        img  = img.crop(bbox)

    w, h   = img.size
    aspect = h/w
    height = np.ceil(aspect * width * vscale).astype(int) # Integer
    height = height if height % 2 == 0 else height + 1 #Make even. important

    if colorful:
        img  = img.resize((width, height)).convert('RGB')
        data = np.reshape(img.getdata(),(height,width,-1))[...,:3]
        data = 225 - data if invert else data #Inversion
        fd   = data[:-1:2,...] #Foreground
        bd   = data[1::2,...]  # Background
        # Upper half block is forground and lower part is background, so one spot make two pixels. 
        d_str  = "\033[48;2;{};{};{}m\033[38;2;{};{};{}m\u2580\033[00m" #Upper half block
        pixels = [[d_str.format(*v1,*v2) for v1,v2 in zip(b,f)] for b,f in zip(bd,fd)]
        
    else:
        height = int(height/2) #
        chars  = ['.',':',';','+','*','?','%','S','#','@']
        chars  = chars[::-1] if invert else chars #Inversion
        img    = img.resize((width, height)).convert('L') # grayscale
        pixels = [chars[int(v*len(chars)/255) -1] for v in img.getdata()]
        pixels = np.reshape(pixels,(height,-1)) #Make row/columns
    
    out_str = '\n'.join([''.join([p for p in ps]) for ps in pixels])

    if outfile:
        with open(outfile,'w') as f:
            f.write(out_str)
    else:
        # For loop is important for printing lines, otherwise breaks appear.
        for line in out_str.splitlines():
            print(line)
