#!/usr/bin/env python
# coding: utf-8

# # Thiết lập môi trường và dataset

# In[37]:


#import library
import numpy as np
import pandas as pd
from tkinter import *
from tkinter import filedialog
# In[38]:
data = pd.read_csv("spam_SMS.csv",encoding='latin1')
data.head()


# # Tiền xử lý dữ liệu

# In[39]:


# Thay đổi tên nhãn "ham" -> 0, "spam" -> 1
#data['class'] = data['class'].replace(['ham','spam'],[0,1])
#data.head()


# In[40]:


data.shape


# In[41]:


y = data['class'].values
X_text = data['content'].values 
print(X_text.shape)
print(y.shape)


# In[42]:


# Lowercase the alphabets
data['content'] = data['content'].str.lower()
data.head()


# In[43]:


from sklearn.feature_extraction.text import CountVectorizer
from nltk.corpus import stopwords

# Loại bỏ stopwords
sw = stopwords.words("english")

# count_vect = CountVectorizer(stop_words="english")
count_vect = CountVectorizer(sw)

# Chuyển đổi định dãng text thành vector
tcv = count_vect.fit_transform(X_text)


# In[44]:


from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords

# Loại bỏ stopwords theo công thức TF-IDF
vectorizer = TfidfVectorizer(stop_words=sw,lowercase=True)

# Chuyển đổi định dãng text thành vector
X = vectorizer.fit_transform(X_text).toarray()

print(X.shape)
print(y.shape)


# In[45]:


# Chia bộ dataset thành training set và testing set
from sklearn.model_selection import train_test_split

# Training set / Testing set : 70% / 30%
# random_state: đảm bảo mỗi lần split đều ra output giống nhau

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=50)
print("X_train lenght:",len(X_train))
print("X_test lenght:",len(X_test))
print("y_train lenght:",len(y_train))
print("y_test lenght:",len(y_test))


# ## Visualize classification report

# In[46]:


# Plot classification report
import matplotlib.pyplot as plt
import numpy as np
import itertools

def plot_classification_report(classificationReport,
                               title='Classification report',
                               cmap='RdBu'):

    classificationReport = classificationReport.replace('\n\n', '\n')
    classificationReport = classificationReport.replace(' / ', '/')
    lines = classificationReport.split('\n')

    classes, plotMat, support, class_names = [], [], [], []
    for line in lines[1:]:  # if you don't want avg/total result, then change [1:] into [1:-1]
        t = line.strip().split()
        if len(t) < 2:
            continue
        classes.append(t[0])
        v = [float(x) for x in t[1: len(t) - 1]]
        support.append(int(t[-1]))
        class_names.append(t[0])
        plotMat.append(v)

    plotMat = np.array(plotMat)
    xticklabels = ['Precision', 'Recall', 'F1-score']
    yticklabels = ['{0} ({1})'.format(class_names[idx], sup)
                   for idx, sup in enumerate(support)]

    plt.imshow(plotMat, interpolation='nearest', cmap=cmap, aspect='auto')
    plt.title(title)
    plt.colorbar()
    plt.xticks(np.arange(3), xticklabels, rotation=45)
    plt.yticks(np.arange(len(classes)), yticklabels)

    upper_thresh = plotMat.min() + (plotMat.max() - plotMat.min()) / 10 * 8
    lower_thresh = plotMat.min() + (plotMat.max() - plotMat.min()) / 10 * 2
    for i, j in itertools.product(range(plotMat.shape[0]), range(plotMat.shape[1])):
        plt.text(j, i, format(plotMat[i, j], '.2f'),
                 horizontalalignment="center",
                 color="white" if (plotMat[i, j] > upper_thresh or plotMat[i, j] < lower_thresh) else "black")

    plt.ylabel('Classes')
    plt.xlabel('Types of Accuracy')
    plt.tight_layout()


# ## Logistic Regression:


class App:
# In[47]:
    def __init__(self, master):
        frame = Frame(master)
        frame.grid()
        window.title("Welcome to Pikachu app")
        self.lbl = Label(window, text="Chao Mung Den Voi Nhom Chung Minh")
        self.lbl.grid(column=3, row=0)
        self.txt = Entry(window, width=30)
        self.txt.grid(column=3, row=1)
        self.Logistic_Regression_butt = Button(frame, text="Logistic Regression", command=self.LogisticRegression)
        self.Logistic_Regression_butt.grid(column=2, row=2)
        self.Gauss_Naive_Bayes_butt = Button(frame, text="Gaussian Naive Bayes", command=self.GaussianNB)
        self.Gauss_Naive_Bayes_butt.grid(column=2, row=3)
        self.Multinomial_Naive_Bayes_butt = Button(frame, text="Multinomial Naive Bayes", command=self.MultinomialNB)
        self.Multinomial_Naive_Bayes_butt.grid(column=2, row=4)
        self.Bernoulli_Naive_Bayes_butt = Button(frame, text="Bernoulli Naive Bayes", command=self.BernoulliNB)
        self.Bernoulli_Naive_Bayes_butt.grid(column=4, row=2)
        self.Decision_tree_butt = Button(frame, text="Decision tree", command=self.tree)
        self.Decision_tree_butt.grid(column=4, row=3)
        self.Support_Vector_Machine_butt = Button(frame, text="Support Vector Machine (SVM)", command=self.SVC)
        self.Support_Vector_Machine_butt.grid(column=4, row=4)
        self.inputtext = self.txt.get()
        self.inputtext_butt=Button(frame,text="Input text",command=self.inputtext)
        self.inputtext_butt.grid(column=3, row=1)
    def LogisticRegression(self):
        from sklearn.linear_model import LogisticRegression
        from sklearn.metrics import accuracy_score
        classifier = LogisticRegression(solver='lbfgs')
        classifier.fit(X_train,y_train)
        pred = classifier.predict(X_test)
        print("Accuracy score",accuracy_score(y_test,pred))
        print("\n")
        from sklearn.metrics import classification_report
        print(classification_report(y_test, pred))
        classifier = LogisticRegression(solver='lbfgs')
        classifier.fit(tcv, data['class'])

        # sentence = self.inputtext
        print("Predicted class:", classifier.predict(count_vect.transform([self.inputtext]))[0])
        def main():
            sampleClassificationReport = """precision    recall  f1-score   support
              Ham       0.96      1.00      0.98      1448
              Spam      0.99      0.70      0.82       225
            avg / total       0.96      0.96      0.96      1673"""

            plot_classification_report(sampleClassificationReport)
            plt.show()
            plt.close()

        if __name__ == '__main__':
            main()


    # In[48]:





    # ## Gaussian Naive Bayes

    # In[49]:

    def GaussianNB(self):
        from sklearn.naive_bayes import GaussianNB
        from sklearn.metrics import accuracy_score
        classifier = GaussianNB()
        classifier.fit(X_train,y_train)
        pred = classifier.predict(X_test)
        print("Accuracy score", accuracy_score(y_test, pred))
        print("\n")
        from sklearn.metrics import classification_report
        print(classification_report(y_test, pred))
        classifier = GaussianNB(solver='lbfgs')
        classifier.fit(tcv, data['class'])

        print("Predicted class:", classifier.predict(count_vect.transform([self.inputtext]))[0])
        def main():

            sampleClassificationReport = """precision    recall  f1-score   support
    
              Ham       0.98      0.91      0.94      1448
              Spam      0.59      0.88      0.71       225
            avg / total       0.93      0.90      0.91      1673"""

            plot_classification_report(sampleClassificationReport)
            plt.show()
            plt.close()

        if __name__ == '__main__':
            main()
    # Predict class example


    # ## Multinomial Naive Bayes

    # In[50]:
    def MultinomialNB(self):
        from sklearn.naive_bayes import MultinomialNB
        from sklearn.metrics import accuracy_score
        classifier = MultinomialNB()
        classifier.fit(X_train,y_train)
        pred = classifier.predict(X_test)
        print("Accuracy score",accuracy_score(y_test,pred))
        print("\n")
        from sklearn.metrics import classification_report
        print(classification_report(y_test, pred))
        classifier = MultinomialNB()
        classifier.fit(tcv, data['class'])

        print("Predicted class:", classifier.predict(count_vect.transform([self.inputtext]))[0])


    # In[51]:


    # Predict class example



    # # Bernoulli Naive Bayes

    # In[52]:
    def BernoulliNB(self):
        from sklearn.naive_bayes import BernoulliNB
        from sklearn.metrics import accuracy_score
        classifier = BernoulliNB()
        classifier.fit(X_train,y_train)
        pred = classifier.predict(X_test)
        print("Accuracy score",accuracy_score(y_test,pred))
        print("\n")
        from sklearn.metrics import classification_report
        print(classification_report(y_test, pred))


    # In[53]:


    # Predict class example
        classifier = BernoulliNB()
        classifier.fit(tcv, data['class'])

        sentence = input(self.inputtext)
        print("Predicted class:",classifier.predict(count_vect.transform([sentence]))[0])


    # ## Decision tree

    # In[54]:
    def tree(self):
        from sklearn import tree
        from sklearn.metrics import accuracy_score
        classifier = tree.DecisionTreeClassifier()
        classifier.fit(X_train,y_train)
        pred = classifier.predict(X_test)
        print("Accuracy score",accuracy_score(y_test,pred))
        print("\n")
        from sklearn.metrics import classification_report
        print(classification_report(y_test, pred))
        classifier = tree.DecisionTreeClassifier()
        classifier.fit(tcv, data['class'])

        print("Predicted class:", classifier.predict(count_vect.transform([self.inputtext]))[0])

        def main():

            sampleClassificationReport = """precision    recall  f1-score   support
    
              Ham       0.99      0.98      0.98      1448
              Spam       0.90      0.91      0.90       225
            avg / total       0.97      0.97      0.97      1673"""

            plot_classification_report(sampleClassificationReport)
            plt.show()
            plt.close()

        if __name__ == '__main__':
            main()


    # In[55]:


    # Predict class example



    # ## Support Vectot Machine (SVM)

    # In[56]:
    def SVC(self):
        from sklearn.svm import SVC
        from sklearn.metrics import accuracy_score
        classifier = SVC(gamma=0.1,C=1,kernel='rbf')
        classifier.fit(X_train,y_train)
        pred = classifier.predict(X_test)
        print("Accuracy score",accuracy_score(y_test,pred))
        print("\n")
        from sklearn.metrics import classification_report
        print(classification_report(y_test, pred))
        classifier = SVC(gamma=0.1, C=1, kernel='rbf')
        classifier.fit(tcv, data['class'])

        print("Predicted class:", classifier.predict(count_vect.transform([self.inputtext]))[0])
        def main():

            sampleClassificationReport = """precision    recall  f1-score   support
    
              Ham       0.95      1.00      0.98      1448
              Spam       0.99      0.68      0.80       225
            avg / total       0.96      0.96      0.95      1673"""

            plot_classification_report(sampleClassificationReport)
            plt.show()
            plt.close()

        if __name__ == '__main__':
            main()


    # In[57]:


    # Predict class example



window = Tk()
App(window)
window.mainloop()
# In[ ]:




