from io import BytesIO
import base64
import PIL
from IPython.display import display, Image, HTML
from PIL import ImageFont
from PIL import ImageDraw 

def get_base64_from_image(img):
    imgbuffer = BytesIO()
    img.save(imgbuffer, 'png')
    return base64.b64encode(imgbuffer.getvalue())

def build_html_images(arrays, scale, show_values=True):
    '''
    Monta a string para mostrar as imagens de uma camada em HTML
    * arrays: Deve ser um array de 3 dimensões no formato (filtros, img_height, img_width)
    * scale: escala que a imagem deve ser aumentada
    * show_values: se True mostra os valores do vetor na imagens construida
    '''
    
    #converte para valores de 0 a 255 uint8
    n_filters,h,w = arrays.shape
    dif = arrays.max()-arrays.min()
    # Evita divisão por zero
    if dif == 0:
        dif = 1
    np_img = np.copy(255*(arrays-arrays.min())/dif)
    np_img = np_img.astype('uint8')
    
    html_images = []
    for f in range(n_filters):
        # cria a imagem no PIL
        img = PIL.Image.fromarray(np_img[f,:,:])
        img = img.resize((w*scale, h*scale))

        if show_values:
            img = img.convert('RGB')

            draw = ImageDraw.Draw(img)
            for i in range(h):
                for j in range(w):
                    draw.text((j*scale, i*scale), 
                              "{:.3f}".format(float(arrays[f,i,j])).rstrip('0').rstrip('.'), 
                              font=ImageFont.load_default(), 
                              fill=(255, 0, 0))
        
        html_images.append("<img src='data:image/png;base64,{}'/>".format(get_base64_from_image(img).decode()))
    
    return html_images

def show_deep_net(model, inputs, scale=1):
    table = "<table><tr><td>Imagem de entrada</td>"
    img = build_html_images(inputs[0,:,:,:], scale)[0]
    table += "<td colspan=100>{}</td>".format(img)
    table += "</tr>"

    # Resultados para cada camada
    i = 1
    for layer in model.layers:
        intermediate_layer_model = Model(inputs=model.input,outputs=layer.output)
        intermediate_output = intermediate_layer_model.predict(inputs)
        table += "<tr><td>Camada {}: {}</td>".format(i, layer.name)
        if len(intermediate_output.shape) == 4:
            for img in build_html_images(intermediate_output[0,:,:,:], scale):
                table += "<td>{}</td>".format(img)
        else:
            n, x = intermediate_output.shape
            for img in build_html_images(intermediate_output.reshape(1,1,-1), scale):
                table += "<td colspan=100>{}</td>".format(img)

        i+=1

        table += "</tr>"
    table += "</table>"
    display(HTML(table))
    