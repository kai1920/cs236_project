from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def evaluation(model, dataset,threshold=0.5):
    probabilities = torch.sigmoid(model(dataset.data))
    y_pred = (probabilities >= threshold).int()
    y_test = dataset.labels
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    conf_mat = confusion_matrix(y_test, y_pred)
    conf_mat = pd.DataFrame(conf_mat,
                              index=['Actual Negative', 'Actual Positive'],
                              columns=['Predicted Negative', 'Predicted Positive'])


    fpr, tpr, thresholds = roc_curve(y_test, probabilities.detach().numpy())
    roc_auc = auc(fpr, tpr)

    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    plt.show()

    print(f"Accuracy: {accuracy}")
    print(f"Precision: {precision}")
    print(f"Recall: {recall}")
    print(f"F1 Score: {f1}")
    print(f"Confusion Matrix: \n{conf_mat}")

import tensorflow as tf
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import numpy as np

class Logger:
    def __init__(self, log_dir):
        self.writer = tf.summary.create_file_writer(log_dir)

    def log_parameters(self, params, prefix='parameters'):
        with self.writer.as_default():
            for key, value in params.items():
                tf.summary.text(f"{prefix}/{key}", str(value))

    def log_training(self, loss, q_value, reward_list, episode):
        mean_reward = np.mean(reward_list)
        std_reward = np.std(reward_list)
        mean_loss = np.mean(loss)
        mean_q = np.mean(q_value)
        with self.writer.as_default():
            tf.summary.scalar('Training Mean Reward', mean_reward, step=episode)
            tf.summary.scalar('Training Std Reward', std_reward, step=episode)
            tf.summary.scalar('Loss', mean_loss, step=episode)
            tf.summary.scalar('Q_mean', mean_q, step=episode)

    def log_evaluation(self, model,reward_list, episode, y_true, y_pred):
        total_reward = np.sum(reward_list)
        std_reward = np.std(reward_list)
        with self.writer.as_default():
            tf.summary.scalar(f'{model}Evaluation Total Reward', total_reward, step=episode)
            tf.summary.scalar(f'{model}Evaluation Std Reward', std_reward, step=episode)

            # Calculate precision, recall, F1 score, and confusion matrix
            precision = precision_score(y_true, y_pred)
            recall = recall_score(y_true, y_pred)
            f1 = f1_score(y_true, y_pred)
            cm = confusion_matrix(y_true, y_pred)
            TN, FP, FN, TP = cm.ravel()

            result_string = f"True Negatives: {TN}, False Positives: {FP}, False Negatives: {FN}, True Positives: {TP}"
            tf.summary.scalar(f'{model}Evaluation Precision', precision, step=episode)
            tf.summary.scalar(f'{model}Evaluation Recall', recall, step=episode)
            tf.summary.scalar(f'{model}Evaluation F1 Score', f1, step=episode)
            tf.summary.text(f'{model}Evaluation Confusion Matrix', result_string, step=episode)

    def close(self):
        self.writer.close()

import emoji
import re

MAX_UNK_PERCENTAGE = 20  # 50% threshold
MIN_LENGTH = 15
# def replace_emojis(text, replacement='[UNK]'):
#     # Define a regular expression pattern for emojis
#     emoji_pattern = re.compile("[" + re.escape("".join(emoji.UNICODE_EMOJI['en'])) + "]")

#     # Replace emojis with the specified replacement
#     return emoji_pattern.sub(replacement, text)

def keep_first_num_only(text):
    # Find all occurrences of [NUM]
    num_occurrences = re.findall(r'\[NUM\]', text)

    # If more than one [NUM] is found, replace all but the first
    if len(num_occurrences) > 1:
        first_num = True

        def replace_num(match):
            nonlocal first_num
            if first_num:
                first_num = False
                return match.group()
            else:
                return ""

        text = re.sub(r'\[NUM\]', replace_num, text)

    return text

def preprocess_text(text):
    # 1. Limit the length to 200 characters
    # text = text

    # 2. Regular expression pattern to keep English letters, digits, and common punctuation marks, replacing others
    # Replace URLs with [URL]
    text = re.sub(r'http\S+|www\S+|https\S+', '[URL]', text, flags=re.MULTILINE)
    pattern = r'[^\x00-\x7F]|[^\w\s.,?/+[\]()\'";:!$#%&*-_]'
    text = re.sub(pattern, '[UNK]', text)

    # 3. Trim whitespace
    text = re.sub(r'\s+', ' ', text)
    text = text.strip()

    # 4. Replace URL and Number
    # Replace numbers with [NUM]
    text = re.sub(r'\b\d+\b', '[NUM]', text)
    text = keep_first_num_only(text)

    origin_text = ''

    while origin_text != text:
      origin_text = text
      # Replace continuous [UNK] occurrences with a single [UNK]
      text = re.sub(r'(\[UNK\])\1+', r'\1', text)

      # Remove [UNK] between letters
      text = re.sub(r'(?<=\w)\[UNK\](?=\w)', '', text)

      # Replace continuous or space-separated [UNK] occurrences with a single [UNK]
      text = re.sub(r'(\[UNK\])(\W*\1)+', r'\1', text)


    # # Replace continuous [UNK] occurrences with a single [UNK]
    # text = re.sub(r'(\[NUM\])\1+', r'\1', text)

    # # Remove [UNK] between letters
    # text = re.sub(r'(?<=\w)\[NUM\](?=\w)', '', text)

    # # Replace continuous or space-separated [UNK] occurrences with a single [UNK]
    # text = re.sub(r'(\[NUM\])(\W*\1)+', r'\1', text)

    return text

def percentage_of_unk(text):
    word_list = text.split()
    # print(word_list)
    total_words = len(text)
    if total_words == 0:
        return 0
    unk_count = text.count('[UNK]')*5
    num_count = text.count('[NUM]')*5
    # print(unk_count, total_words)
    return ((unk_count+num_count) / total_words) * 100

def is_acceptable_by_percentage(text, max_unk_percentage=MAX_UNK_PERCENTAGE):
    cnt_word = len(text.split(' '))
    return (percentage_of_unk(text) <= max_unk_percentage) and (cnt_word>=MIN_LENGTH)

def initial_data_cleanup(my_data,max_unk_percentage=MAX_UNK_PERCENTAGE):
    my_data['original_text'] = my_data['payload_text']
    my_data['payload_text'] = my_data['payload_text'].apply(preprocess_text)
    my_data['is_acceptable'] = my_data['payload_text'].apply(is_acceptable_by_percentage)
    my_data = my_data[my_data['is_acceptable']]
    return my_data


