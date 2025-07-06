import tensorflow
import numpy as np
from PIL import Image
import cv2
import streamlit as st
from uform import get_model, Modality
from deepface import DeepFace
from zipfile import ZipFile
import zipfile
from pathlib import Path
import imageio,cv2
import io

@st.cache_data
def get_face_embedding(faces, names):
    d, fs = [], []
    bytes_data = faces.getvalue()
    img = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    try: 
        faces_embedding = DeepFace.represent(img, model_name = 'Facenet')
        for loop1, face_embedding in enumerate(faces_embedding):
            x,y,w,h,_,_ = face_embedding['facial_area'].values()
            fs.append([x,y,w,h])
            for loop2, name in enumerate(names):
                for n in range(len(name[:-1])):
                    try:
                        ref_embedding = DeepFace.represent(name[n], model_name = 'Facenet')
                    except Exception as e:
                        continue
                    else:
                        for ref in ref_embedding:
                            f = face_embedding['embedding']
                            r = ref['embedding']
                            sim = round(float(np.dot(f, r))/(np.linalg.norm(f)*np.linalg.norm(r)), 2)
                            d.append([sim, name[n], loop1, n + len(names[max(0,loop2-1)])*min(1,loop2) - 1])

    except Exception as e:
        st.error(f"ERROR: {str(e)}")
    
    return d, fs
    
@st.cache_data
def prepare_data(source,text):
    processors, models = get_model('unum-cloud/uform3-image-text-english-small')
    model_text = models[Modality.TEXT_ENCODER]
    model_image = models[Modality.IMAGE_ENCODER]
    processor_text = processors[Modality.TEXT_ENCODER]
    processor_image = processors[Modality.IMAGE_ENCODER]
    info = []
    for images in source:
        for i in images[:-1]:
            image = Image.open(i)
            image_data = processor_image(image)
            _, image_embedding = model_image.encode(image_data)
            info.append([image_embedding, i])
    
    text_data = processor_text(text)
    _, text_embedding = model_text.encode(text_data)
    return info, text_embedding

def text_search(info, text_embedding):
    return round(float(np.dot(info.flatten(), text_embedding.reshape(-1)))/(np.linalg.norm(info.flatten())*np.linalg.norm(text_embedding.reshape(-1))), 2)

def zip(file):
    source = []
    data = []
    for f in file:
        with ZipFile(f, 'r') as zip:
            zip.extractall()
            name = zip.namelist()
            for n in name:
                with zip.open(n) as z:
                    data.append(z.read())
            name.append(f.name)
        source.append(name)
    return source, data

def visualize():
    with st.sidebar:
        file = st.file_uploader('Upload Images', type = 'ZIP', accept_multiple_files= True)
        if file: 
            source, data = zip(file) 
    st.title('🗿Image Search')
    if not file:
        st.text('👈 Please upload your images')
    else:
        with st.expander('Source Images'):
            bool1 = False  
            count0 = 0
            colu1,colu2,colu3 = st.columns(3)
            for i in source:
                for ii in i[:-1]:
                    try:                            
                        count0 += 1
                        if count0 in list(range(1, len(source[0])*len(source) + 1, 3)):
                            with colu1: st.image(ii, width = 150, caption = f'{Path(ii).stem} from {i[-1]}')
                        elif count0 in list(range(2, len(source[0])*len(source) + 2, 3)):
                            with colu2: st.image(ii, width = 150, caption = f'{Path(ii).stem} from {i[-1]}')
                        else:
                            with colu3: st.image(ii, width = 150, caption = f'{Path(ii).stem} from {i[-1]}')
                    except: 
                        bool1 = True
                        count0 -= 1
            if bool1: 
                st.warning("WARNING: Your ZIP file contains some non-image files.")
        t1,t2 = st.tabs(['🙂 Face Search', '📄 Text Search'])
        with t1:
            col1, col2 = st.columns(2)
            with col1:
                photo = st.camera_input("Take a picture")
            with col2:
                image = st.file_uploader("Upload Image", type = ['PNG','JPG', 'JPEG', 'WEBP'])
                if image: st.image(image, width = 300)

            sim1 = st.slider("Level of Similarity(%)",key = 1,  min_value = 10, max_value = 99, value = 70)   
            if photo and image: 
                st.error('ERROR: Only one at a time!')
                return
            elif not photo and not image: return
            else:
                input_img = photo if photo != None else image
                if input_img: 
                    d, faces = get_face_embedding(input_img, source)
            co1, co2, co3 = st.columns(3)
            b = False 
            count1 = 0                      
            if d:
                zip_buffer1 = io.BytesIO()   
                with ZipFile(zip_buffer1, "w", zipfile.ZIP_DEFLATED) as zip_file:
                    for i in d:
                        if i[0]*100 >= sim1: 
                            count1 += 1
                            if count1 in list(range(1, len(d) + 1, 3)): 
                                with co1: st.image(i[1], width = 150, caption = f"{int(i[0]*100)}% compared to input image {i[2]+1}")
                            elif count1 in list(range(2, len(d) + 2, 3)): 
                                with co2: st.image(i[1], width = 150, caption = f"{int(i[0]*100)}% compared to input image {i[2]+1}")
                            else: 
                                with co3: st.image(i[1], width = 150, caption = f"{int(i[0]*100)}% compared to input image {i[2]+1}")
                            zip_file.writestr(i[1], data[i[-1]])
                            b = True
                with st.sidebar:
                    for num, face in enumerate(faces):
                        x,y,w,h = face
                        bytes_data = input_img.getvalue()
                        nparr = np.frombuffer(bytes_data, np.uint8)
                        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                        img = img[y:y+h,x:w+x]
                        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                        st.image(img_rgb, width = 100, caption = f"Input image {num + 1}")
                if b:
                    if st.download_button("Tải về file ZIP", data=zip_buffer1.getvalue(), file_name="found_images.zip", key = '10'):
                        st.success('Downloaded Images')

        with t2:
            c1,c2 = st.columns(2)
            with c1:
                text = st.text_area('Image Description')
            with c2:
                sim2 = st.slider("Level of Similarity(%)", key = 2, min_value = 10, max_value = 99, value = 70)
            if text:
                info, text_embedding = prepare_data(source, text)
                c_1,c_2,c_3 = st.columns(3)
                count2 = 0
                zip_buffer2 = io.BytesIO()   
                with ZipFile(zip_buffer2, "w", zipfile.ZIP_DEFLATED) as zip_file:
                    c = False
                    k = 0
                    for i in info:
                        s = text_search(np.array(i[0]), np.array(text_embedding))*100
                        if s >= sim2:
                            count2 += 1
                            if count2 in list(range(1, len(info) + 1, 3)):
                                with c_1: st.image(i[1], width = 200, caption = f"{int(s)}%")
                            elif count2 in list(range(2, len(info) + 2, 3)):
                                with c_2: st.image(i[1], width = 200, caption = f"{int(s)}%")
                            else:
                                with c_3: st.image(i[1], width = 200, caption = f"{int(s)}%")
                            zip_file.writestr(i[1], data[k])
                            c = True
                        k += 1
                if c:
                    if st.download_button("Tải về file ZIP", data=zip_buffer2.getvalue(), file_name="found_images.zip", key = '11'):
                        st.success('Downloaded Images')
                        st.balloons()
                
visualize()