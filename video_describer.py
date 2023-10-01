import pandas as pd
import re

from video2text import video_dataframe
from tqdm import tqdm
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from torch.utils.data import Dataset, DataLoader
from nltk.translate import meteor
from nltk.translate.bleu_score import sentence_bleu
from nltk.tokenize import word_tokenize
from nltk.translate.nist_score import sentence_nist
from collections import Counter
from transformers import BertModel, BertTokenizer
from transformers import GPT2Tokenizer, T5ForConditionalGeneration
import cv2
import torch
from torchvision import models, transforms
from torchvision.models.resnet import ResNet50_Weights
from PIL import Image
import os
import numpy as np
import shutil
import pickle

with open('imagenet_classes_ru.pickle', 'rb') as handle:
    imagenet_classes_ru = pickle.load(handle)


def is_overlapping(box1, box2):
    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2
    return not (x1 + w1 < x2 or x2 + w2 < x1 or y1 + h1 < y2 or y2 + h2 < y1)


def union_contours(contours):
    new_contours = []
    skip = set()
    for i in range(len(contours)):
        if i in skip:
            continue
        contour1 = contours[i]
        box1 = cv2.boundingRect(contour1)
        union = None
        for j in range(i + 1, len(contours)):
            contour2 = contours[j]
            box2 = cv2.boundingRect(contour2)
            if is_overlapping(box1, box2):
                skip.add(j)
                if union is None:
                    union = np.concatenate((contour1, contour2))
                else:
                    union = np.concatenate((union, contour2))
        if union is None:
            new_contours.append(contour1)
        else:
            new_contours.append(union)
    return new_contours


def process_images(image_folder, output_folder, min_contour_area=0):
    print('Processing images...')
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for image_file in os.listdir(image_folder):
        image_path = os.path.join(image_folder, image_file)
        image = cv2.imread(image_path)
        img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        img_gray = cv2.GaussianBlur(img_gray, (5, 5), 0)

        mean_val = np.mean(img_gray)
        std_dev = np.std(img_gray)

        lower_threshold = mean_val - std_dev
        upper_threshold = mean_val + std_dev

        edges = cv2.Canny(img_gray, lower_threshold, upper_threshold * 4.5)
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours = union_contours(contours)

        for index, contour in enumerate(contours):
            x, y, w, h = cv2.boundingRect(contour)
            if w * h < min_contour_area:
                continue
            cropped_img = image[y:y + h, x:x + w]
            output_path = os.path.join(output_folder, f"{os.path.splitext(image_file)[0]}_parts_{index}.jpg")
            cv2.imwrite(output_path, cropped_img)

        shutil.copy(image_path, os.path.join(output_folder, f"{os.path.basename(image_path)}"))

    print('Processing images - DONE')


def classify_images(folder_path, folder_number):
    print('Images classification...')
    model3 = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
    model3.eval()

    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    predictions = []

    for filename in os.listdir(folder_path):
        if filename.endswith(('.jpg', '.jpeg', '.png')):
            file_path = os.path.join(folder_path, filename)

            input_image = Image.open(file_path)
            input_tensor = preprocess(input_image)
            input_batch = input_tensor.unsqueeze(0)

            with torch.no_grad():
                output = model3(input_batch)

            _, predicted = torch.max(output, 1)
            class_name = imagenet_classes_ru.get(predicted.item())
            predictions.append(class_name)

    predictions_variable_name = f'predictions_{folder_number}'
    output_filename = f'Objects_{folder_number}.txt'
    with open(output_filename, 'w') as file:
        for prediction in predictions:
            file.write(f'{prediction}\n')

    # print('Images classification - DONE')
    return predictions


def process_video(video_path, threshold=0.96):
    # print('Processing video...')
    cap = cv2.VideoCapture(video_path)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    prev_hist = None
    count = 0

    frames_dir = video_path.replace('.mp4', '')
    if not os.path.exists(frames_dir):
        os.makedirs(frames_dir)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if count % fps == 0:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
            cv2.normalize(hist, hist, 0, 1, cv2.NORM_MINMAX, -1)

            if count == 0:
                frame_filename = f'frame_0.jpg'
                frame_filepath = os.path.join(frames_dir, frame_filename)
                cv2.imwrite(frame_filepath, frame, [cv2.IMWRITE_JPEG_QUALITY, 70])

            elif prev_hist is not None:
                correlation = cv2.compareHist(prev_hist, hist, cv2.HISTCMP_CORREL)
                if correlation < threshold:
                    frame_filename = f'frame_{int(count / fps)}.jpg'
                    frame_filepath = os.path.join(frames_dir, frame_filename)
                    cv2.imwrite(frame_filepath, frame, [cv2.IMWRITE_JPEG_QUALITY, 70])

            prev_hist = hist
        count += 1

    cap.release()
    # print('Processing video - DONE')


def video_dataframe(path2v, video_number, threshold=0.96, max_rectangle=10000):
    video_path = f'{path2v}{video_number}'
    images_folder = f'{path2v}{video_number}'.replace(".mp4", '')
    parts_folder = f'{path2v}{video_number}_parts'.replace(".mp4", '')

    process_video(video_path, threshold)
    process_images(images_folder, parts_folder, max_rectangle)
    data_info = classify_images(parts_folder, video_number)

    shutil.rmtree(parts_folder)
    shutil.rmtree(images_folder)

    return data_info


def del_timestamps(text):
    text = text.split("] ")[1:]
    return " ".join(text)


def ret_stt(stt_name):
    with open(f"./train_stt/{stt_name}", 'r', encoding="utf_8_sig") as f:
        lines = f.readlines()
        lines = [del_timestamps(line.strip()) for line in lines]
    return lines


def process_corpus_(data, corpus_index):
    data["stt"] = data["stt_name"].apply(ret_stt)
    txt = data["stt"][corpus_index]

    tmp = []
    flag = 0

    for i in txt:

        for j in i:
            if j == '*':
                flag = 1
            else:
                continue
        if flag == 0 and len(i.split()) > 3:
            tmp.append(i)
        else:
            flag = 0

    return pd.Series(tmp)


def process_corpus(file_name):
    data = pd.read_csv(file_name)
    data = data.head(20)
    data["stt_sum"] = [process_corpus_(data, i) for i in range(data.shape[0])]
    return data


tokenizer = BertTokenizer.from_pretrained('content/rubert_cased_L-12_H-768_A-12_pt')
model = BertModel.from_pretrained('content/rubert_cased_L-12_H-768_A-12_pt', output_hidden_states=True)
device = torch.device("cpu")
model.to(device)
model.eval()
tokenizer1 = GPT2Tokenizer.from_pretrained('ai-forever/FRED-T5-1.7B', eos_token='</s>')
model1 = T5ForConditionalGeneration.from_pretrained('ai-forever/FRED-T5-1.7B')
device = 'cpu'
model1.to(device)


class CustomDataset(Dataset):

    def __init__(self, X):
        self.text = X

    def tokenize(self, text):
        return tokenizer(text, return_tensors='pt', padding='max_length', truncation=True, max_length=150)

    def __len__(self):
        return self.text.shape[0]

    def __getitem__(self, index):
        output = self.text[index]
        output = self.tokenize(output)
        return {k: v.reshape(-1) for k, v in output.items()}


def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output['last_hidden_state']
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
    sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
    return sum_embeddings / sum_mask


def top_words_list(words):
    word_counts = Counter(words)
    top_10_words = word_counts.most_common(min(10, len(word_counts)))

    only_words = [word for word, count in top_10_words]

    return only_words


def calc(corpus, video_name, path_2_video):
    eval_ds = CustomDataset(corpus)
    eval_dataloader = DataLoader(eval_ds, batch_size=10)
    if len(corpus) > 50:

        embeddings = torch.Tensor().to(device)

        with torch.no_grad():
            for n_batch, batch in enumerate(tqdm(eval_dataloader)):
                batch = {k: v.to(device) for k, v in batch.items()}
                outputs = model(**batch)
                embeddings = torch.cat([embeddings, mean_pooling(outputs, batch['attention_mask'])])
            embeddings = embeddings.cpu().numpy()

        pca = PCA(n_components=15, random_state=42)
        emb_15d = pca.fit_transform(embeddings)

        kmeans = KMeans(n_clusters=10, random_state=42)
        cluster_labels = kmeans.fit_predict(emb_15d)
        cluster_centers = kmeans.cluster_centers_
        unique_clusters = np.unique(cluster_labels)

        cluster_centers_indices = {}
        for cluster_label in unique_clusters:
            cluster_centers_indices[cluster_label] = np.where(cluster_labels == cluster_label)[0][0]

        tmp = []
        for i in cluster_centers_indices.values():
            tmp.append(i)
        tmp.sort()

        line1 = ""
        core_sentences = []
        for i in range(len(tmp) // 2):
            line1 = line1 + corpus[tmp[i]] + f" <extra_id_{i}>"
            core_sentences.append(corpus[tmp[i]])

        line2 = ""
        for i in range(len(tmp) // 2, len(tmp)):
            line2 = line2 + corpus[tmp[i]] + f" <extra_id_{i - len(tmp) // 2}>"
            core_sentences.append(corpus[tmp[i]])

        lm_text = "Заполни пробелы: " + line1
        input_ids = torch.tensor([tokenizer1.encode(lm_text)]).to(device)
        outputs = model1.generate(input_ids, eos_token_id=tokenizer1.eos_token_id, early_stopping=True)
        replace_dict = {
            match.group(): replacement
            for match, replacement in zip(re.finditer(r'<extra_id_\d+>', tokenizer1.decode(outputs[0][1:])),
                                          re.split(r'<extra_id_\d+>', tokenizer1.decode(outputs[0][1:]))[1:])
        }

        def replacer(match):
            return replace_dict.get(match.group(), '')

        result1 = re.sub(r'<extra_id_\d+>', replacer, line1)

        lm_text = "Заполни пробелы: " + line2
        input_ids = torch.tensor([tokenizer1.encode(lm_text)]).to(device)
        outputs = model1.generate(input_ids, eos_token_id=tokenizer1.eos_token_id, early_stopping=True)
        replace_dict = {
            match.group(): replacement
            for match, replacement in zip(re.finditer(r'<extra_id_\d+>', tokenizer1.decode(outputs[0][1:])),
                                          re.split(r'<extra_id_\d+>', tokenizer1.decode(outputs[0][1:]))[1:])
        }

        def replacer(match):
            return replace_dict.get(match.group(), '')

        result2 = re.sub(r'<extra_id_\d+>', replacer, line2)

        final_line = result1 + result2

        print(final_line)
        return final_line


    else:
        video_data = video_dataframe(path_2_video, video_name, threshold=0.94, max_rectangle=40000)
        tmp = top_words_list(video_data)
        line1 = ""
        core_sentences = []
        for i in range(len(tmp) // 2):
            line1 = line1 + tmp[i] + f" <extra_id_{i}>"
            core_sentences.append(tmp[i])

        line2 = ""
        for i in range(len(tmp) // 2, len(tmp)):
            line2 = line2 + tmp[i] + f" <extra_id_{i - len(tmp) // 2}>"
            core_sentences.append(tmp[i])

        lm_text = "Заполни пробелы: " + line1
        input_ids = torch.tensor([tokenizer1.encode(lm_text)]).to(device)
        outputs = model1.generate(input_ids, eos_token_id=tokenizer1.eos_token_id, early_stopping=True)
        replace_dict = {
            match.group(): replacement
            for match, replacement in zip(re.finditer(r'<extra_id_\d+>', tokenizer1.decode(outputs[0][1:])),
                                          re.split(r'<extra_id_\d+>', tokenizer1.decode(outputs[0][1:]))[1:])
        }

        def replacer(match):
            return replace_dict.get(match.group(), '')

        result1 = re.sub(r'<extra_id_\d+>', replacer, line1)

        lm_text = "Заполни пробелы: " + line2
        input_ids = torch.tensor([tokenizer1.encode(lm_text)]).to(device)
        outputs = model1.generate(input_ids, eos_token_id=tokenizer1.eos_token_id, early_stopping=True)
        replace_dict = {
            match.group(): replacement
            for match, replacement in zip(re.finditer(r'<extra_id_\d+>', tokenizer1.decode(outputs[0][1:])),
                                          re.split(r'<extra_id_\d+>', tokenizer1.decode(outputs[0][1:]))[1:])
        }

        def replacer(match):
            return replace_dict.get(match.group(), '')

        result2 = re.sub(r'<extra_id_\d+>', replacer, line2)

        final_line = result1 + result2
        print(final_line)
        return final_line


def meteor_metric(text, text_sum):
    if isinstance(text_sum, str):
        return round(meteor([word_tokenize(text)], word_tokenize(text_sum)), 4)
    else:
        return 0


def bleu_metric(reference, hypothesis):
    reference = [word_tokenize(reference)]
    hypothesis = word_tokenize(hypothesis)
    return round(sentence_bleu(reference, hypothesis), 4)


def nist_metric(reference, hypothesis):
    try:
        reference = [word_tokenize(reference)]
        hypothesis = word_tokenize(hypothesis)
        return round(sentence_nist(reference, hypothesis), 4)
    except ZeroDivisionError:
        return 0


path2csv = "train.csv"
path2videos = "train_video/"

data = process_corpus(path2csv)

data["desc_proc"] = data.apply(lambda x: calc(x.stt_sum, x.video_name, path2videos), axis=1)

data["met"] = data.apply(lambda x: meteor_metric(x.description, x.desc_proc), axis=1)
data["nist"] = data.apply(lambda x: nist_metric(x.description, x.desc_proc), axis=1)

print("met: ", data["met"].values)
print("met mean: ", data.met.mean())
print("nist: ", data["nist"].values)
print("nist mean: ", data.nist.mean())
