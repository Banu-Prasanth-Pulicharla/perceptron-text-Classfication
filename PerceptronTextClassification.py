import os
from nltk.stem.snowball import SnowballStemmer

punctuation_str = "!\"#$%&'()*+,-./:;<=>?@[\]^_`{|}~"
stopwords_list = ['i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', "you're", "you've", "you'll", "you'd", 'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she', "she's", 'her', 'hers', 'herself', 'it', "it's", 'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves', 'what', 'which', 'who', 'whom', 'this', 'that', "that'll", 'these', 'those', 'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does', 'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until', 'while', 'of', 'at', 'by', 'for', 'with', 'about', 'against', 'between', 'into', 'through', 'during', 'before', 'after', 'above', 'below', 'to', 'from', 'up',
                  'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again', 'further', 'then', 'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very', 's', 't', 'can', 'will', 'just', 'don', "don't", 'should', "should've", 'now', 'd', 'll', 'm', 'o', 're', 've', 'y', 'ain', 'aren', "aren't", 'couldn', "couldn't", 'didn', "didn't", 'doesn', "doesn't", 'hadn', "hadn't", 'hasn', "hasn't", 'haven', "haven't", 'isn', "isn't", 'ma', 'mightn', "mightn't", 'mustn', "mustn't", 'needn', "needn't", 'shan', "shan't", 'shouldn', "shouldn't", 'wasn', "wasn't", 'weren', "weren't", 'won', "won't", 'wouldn', "wouldn't"]

stemmer = SnowballStemmer("english")


def processing_folder(folder):
    folder = folder.replace("\\", "/")
    list_of_files = os.listdir(folder)
    return list_of_files, folder


def file_list_to_dict(list_of_files, folder):
    email_dict = {}
    for file in list_of_files:
        path = folder+"/"+file

        with open(path, 'r', errors='ignore') as f:
            email_dict[file] = f.read()
    return email_dict


def dict_to_wordcount_dict(email_dict, ignore_stopwords):
    dict_with_word_count = {}
    for email in email_dict:
        email_dict[email] = email_dict[email].split()
        word_list = [''.join(temp for temp in word if temp not in punctuation_str)
                     for word in email_dict[email]]
        word_list = [temp for temp in word_list if temp]

        temp_word_list = []
        for word in word_list:
            temp_word_list.append(stemmer.stem(word))
        if ignore_stopwords == True:
            for sw in stopwords_list:
                if sw in temp_word_list:
                    temp_word_list.remove(sw)

        email_dict[email] = temp_word_list

        for word in temp_word_list:
            if word in dict_with_word_count:
                dict_with_word_count[word] = dict_with_word_count[word] + 1
            else:
                dict_with_word_count[word] = 1
    return dict_with_word_count, email_dict


def calc_weights(num_iterations, eta, train_files_dict, train_wc_dict):

    weights_dict = {'bias': 0.1}
    for word in train_wc_dict:
        weights_dict[word] = 0.0

    for itr in range(num_iterations):
        for file in train_files_dict:
            pred_val = weights_dict['bias']

            temp_dict = {}
            for word in train_files_dict[file]:
                if word in temp_dict:
                    temp_dict[word] = temp_dict[word] + 1
                else:
                    temp_dict[word] = 1

            for word in temp_dict:
                if word in train_wc_dict:
                    pred_val += temp_dict[word] * weights_dict[word]

            if pred_val > 0:
                pred_class = 1
            else:
                pred_class = 0

            if 'spam.txt' in file:
                exep_class = 1
            else:
                exep_class = 0

            # update weights
            for word in temp_dict:
                weights_dict[word] += eta * \
                    (exep_class - pred_class) * temp_dict[word]

    return weights_dict


def get_accuracy(test_email_dict, weights_dict, ignore_stopwords):
    total_count = 0
    correct_count = 0
    for file in test_email_dict:
        pred_val = weights_dict['bias']
        total_count += 1
        temp_dict1 = {}
        for word in test_email_dict[file]:
            if word in temp_dict1:
                temp_dict1[word] = temp_dict1[word] + 1
            else:
                temp_dict1[word] = 1

        for word in temp_dict1:
            if word in weights_dict:
                pred_val += weights_dict[word] * temp_dict1[word]

        if pred_val > 0:
            if 'spam.txt' in file:
                correct_count += 1
        else:
            if 'ham.txt' in file:
                correct_count += 1

    print("Accuracy: ", str(round((correct_count)/(total_count), 2)))


input_value = input(
    "Enter the inputs with spaces: <Training Set Ham Path> <Training Set Spam Path> <Test Set Ham Path> <Test Set Spam Path> <learning Rate>: ")
input_value = input_value.split(' ')


def main(input_value, ignore_stopwords, num_iterations):
    train_ham_inp_folder = input_value[0]
    train_spam_inp_folder = input_value[1]
    test_ham_inp_folder = input_value[2]
    test_spam_inp_folder = input_value[3]
    # num_iterations = int(input_value[4])
    eta = float(input_value[4])

    # Get Ham Data
    train_ham_file_list, train_ham_inp_folder = processing_folder(
        train_ham_inp_folder)

    train_ham_email_dict = file_list_to_dict(
        train_ham_file_list, train_ham_inp_folder)

    train_ham_dict_count, train_ham_email_dict = dict_to_wordcount_dict(
        train_ham_email_dict, ignore_stopwords)

    # Get Spam Data
    train_spam_file_list, train_spam_inp_folder = processing_folder(
        train_spam_inp_folder)

    train_spam_email_dict = file_list_to_dict(
        train_spam_file_list, train_spam_inp_folder)

    train_spam_dict_count, train_spam_email_dict = dict_to_wordcount_dict(
        train_spam_email_dict, ignore_stopwords)

    # combine spam and ham dicts
    train_spam_email_dict.update(train_ham_email_dict)
    train_email_dict = train_spam_email_dict

    for key in train_spam_dict_count:
        if key in train_ham_dict_count:
            train_ham_dict_count[key] += train_spam_dict_count[key]

    train_spam_dict_count.update(train_ham_dict_count)
    train_dict_count = train_spam_dict_count

    # Calculate weights
    weights_dict = calc_weights(
        num_iterations, eta, train_email_dict, train_dict_count)

    # Testing Data
    test_spam_file_list, test_spam_inp_folder = processing_folder(
        test_spam_inp_folder)

    test_spam_email_dict = file_list_to_dict(
        test_spam_file_list, test_spam_inp_folder)

    test_spam_dict_count, test_spam_email_dict = dict_to_wordcount_dict(
        test_spam_email_dict, ignore_stopwords)

    test_ham_file_list, test_ham_inp_folder = processing_folder(
        test_ham_inp_folder)

    test_ham_email_dict = file_list_to_dict(
        test_ham_file_list, test_ham_inp_folder)

    test_ham_dict_count, test_ham_email_dict = dict_to_wordcount_dict(
        test_ham_email_dict, ignore_stopwords)

    # combine spam and ham dicts
    test_spam_email_dict.update(test_ham_email_dict)
    test_email_dict = test_spam_email_dict

    print("-----------------------")
    if ignore_stopwords == True:
        print("Without StopWords (eta = "+str(eta) +
              ", epochs = "+str(num_iterations) + "):")
    else:
        print("With StopWords (eta = "+str(eta) +
              ", epochs = "+str(num_iterations) + "):")
    print("-----------------------")

    get_accuracy(test_email_dict, weights_dict, ignore_stopwords)

# D:\data_TEMP\2\train\ham D:\data_TEMP\2\train\spam D:\data_TEMP\2\test\ham D:\data_TEMP\2\test\spam 0.01


for i in range(1, 21):
    main(input_value, False, i)

for i in range(1, 21):
    main(input_value, True, i)
