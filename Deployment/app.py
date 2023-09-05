import streamlit as st
import re
import spacy
from heapq import nlargest  

st.title("Text Summarizer")
with st.form(key = "clf_form"):
    text_input = st.text_area("Type Here: ")
    input_slider = st.slider(step=0.1,min_value=0.2,max_value=0.7,label="How much portion of the text do you wish to be summarized, Eg: 0.2 --> 20% of the Original Text")
    submit_btn = st.form_submit_button(label = "Submit")
    countOfWords = len(text_input.split())

    class Model():
        try:
            nlp = spacy.load("en_core_web_sm")
        except OSError:
            import subprocess
            subprocess.run(["python", "-m", "spacy", "download", "en_core_web_sm"])
        def predict(text):
            stop_words = [ 'stop', 'the', 'to', 'and', 'a', 'in', 'it', 'is', 'I', 'that', 'had', 'on', 'for', 'were', 'was']
            nlp = spacy.load("en_core_web_sm")
            doc = nlp(text)
    
            lemmatized_text = " ".join([token.lemma_ for token in doc])
    
            re_text = re.sub("[^\s\w,.]"," ",lemmatized_text)
            re_text = re.sub("[ ]{2,}"," ",re_text).lower()
    
            word_frequencies = {}
            for word in doc:
                if word.text not in "\n":
                    if word.text not in stop_words:
                        if word.text not in word_frequencies.keys():
                            word_frequencies[word.text] = 1
                        else:
                            word_frequencies[word.text] +=1
    
            max_word_frequency = max(word_frequencies.values(),default=0)
    
            for word in word_frequencies.keys():
                word_frequencies[word] = word_frequencies[word] / max_word_frequency
    
            sent_tokens = [sent for sent in doc.sents]
            sent_scores = {}
    
            for sent in sent_tokens:
                for word in sent:
                    if word.text in word_frequencies.keys():
                        if sent not in sent_scores.keys():
                            sent_scores[sent] = word_frequencies[word.text]
                        else:
                            sent_scores[sent] += word_frequencies[word.text]
       
            sentence_length = int(len(sent_tokens)*input_slider)
            summary = nlargest(sentence_length,sent_scores,sent_scores.get)
            final_summary = [word.text for word in summary]
            final_summary = " ".join(final_summary)
            return final_summary
    
    
    if submit_btn:
        if text_input == "":
            st.error("Enter something in order to summarize it.",icon="⛔️")
        else:
            if countOfWords <=100:
                st.warning("Pls enter more than 100 words in order to summarize it.",icon="⚠️")
            else:
                st.subheader("Output: ")
            
                col1,col2 = st.columns(2)

                output = Model.predict(text=text_input)
            
                with col1:
                    st.info("Original Text: ")
                    st.write(text_input)

                with col2:
                    st.info("Summarized Text: ")
                    st.write(output)