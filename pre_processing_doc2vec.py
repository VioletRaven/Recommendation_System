from pdfminer.converter import TextConverter
from pdfminer.layout import LAParams
from pdfminer.pdfinterp import PDFResourceManager, PDFPageInterpreter
from pdfminer.pdfpage import PDFPage
from io import StringIO
import string
import nltk
import warnings
import os
import time
import multiprocessing
from gensim.models.doc2vec import TaggedDocument
from tqdm import tqdm
from gensim.models import Doc2Vec
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from os import path

warnings.filterwarnings("ignore")
tqdm.pandas(desc="progress-bar")
nltk.download('stopwords')
nltk.download('punkt')

import pdb

class saving:
    @staticmethod
    def save(path_to_save, object):
        '''
        - Salva in folder preesistente
        - Salva il dataframe o il modello
        - Evita la sovrascrittura dei files se settatato True
        - Ritorno il nuovo percorso
        '''
        i = 0
        path_split = path_to_save.split('.')[0]
        extension = path_to_save.split('.')[1]
        while path.exists(path_split + '_{}'.format(i) + '.' + extension): # crea file più nuovo incrementando di 1 dall'ultimo file caricato
            i += 1
        new_path = path_split + '_{}'.format(i) + '.' + extension
        if extension == 'model':
            print('Saving model...')
            object.save(new_path)
        if extension == 'csv':
            print('Saving .csv file...')
            object.to_csv(new_path)
        return new_path
class cleaner_d2v:
    @staticmethod
    def text_cleaning(text):
        """
        Minima pulitura testuale necessaria all'analisi NLP per la costruzione di embeddings
        * Rimuove spazi e linee a capo
        * Rimuove la punteggiatura
        * Transforma in minuscolo
        * Rimuove caratteri speciali come ’ e “
        * Rimuove le stopwords (parole inutili)
        * Rimuove i numeri
        """
        text = text.replace('\n', '')
        text = text.split()
        table = str.maketrans('', '', string.punctuation)
        stripped = [w.translate(table) for w in text]
        lower_text = [x.lower() for x in stripped]
        lower_text = [x.split("’") for x in lower_text]
        lower_text = [[x.split('“') for x in y] for y in lower_text]
        lower_text = [x for sublist in lower_text for x in sublist]
        new_text = [x for sublist in lower_text for x in sublist]
        new_text = [x for x in new_text if x != r"\x0c"]
        my_stopwords = ["’", "nell’", "d", "l", "’", "'", "“", "", "l°", "-"] # questo carattere è un ’ not a ' --> uguale qui “  "
        stopwords = nltk.corpus.stopwords.words('italian') + nltk.corpus.stopwords.words('english') + my_stopwords # sono presenti anche parole inglesi quindi le includiamo

        new_text = [c for c in new_text if c not in stopwords] # rimuovo le stopwords dal testo
        new_text = [i for i in new_text if not any(char.isdigit() for char in i)] # rimuovo i numeri dal testo perchè non informativi
        final_text = list(filter(None, new_text)) # rimuovo gli spazi vuoti tokenizzati

        return final_text

    @staticmethod
    def convert_to_tuple(path, category_):
        """
        * Converte files pdf in testo e Ritorno una tupla contenente il percorso del file, il testo e la categoria appartenente
        """
        rsrcmgr = PDFResourceManager()
        retstr = StringIO()
        laparams = LAParams()
        device = TextConverter(rsrcmgr, retstr, laparams=laparams)
        fp = open(path, 'rb')
        interpreter = PDFPageInterpreter(rsrcmgr, device)
        password = ""
        maxpages = 0
        caching = True
        pagenos = set()

        for page in PDFPage.get_pages(fp, pagenos, maxpages=maxpages, password=password, caching=caching,
                                      check_extractable=True):
            interpreter.process_page(page)

        text = retstr.getvalue()
        fp.close()
        device.close()
        retstr.close()
        data = (path, text, category_) # ritorno tupla contente il percorso, il testo e la categoria

        return data

    @staticmethod
    def pdf_reader(path_folder):
        '''
        - Apre i files pdf nelle cartelle e li legge tramite il "convert_to_tuple"
        - Salva il dataframe
        '''
        bad_files = 0
        all_files = []
        print('Inizio la lettura...')
        start = time.time()
        for dir in os.listdir(path_folder):
            path = path_folder + r'\\' + dir
            for file in os.listdir(path):
                start_2 = time.time()
                print('Convertendo questo file --> {}'.format(file))
                pdf = path + r'\\' + file
                try:
                    all_files.append(cleaner_d2v.convert_to_tuple(path=pdf, category_=dir))
                except TypeError:
                    bad_files += 1
                    print('Non riesco a leggere questo file')
                    pass

                print('Ci sono voluti {} secondi per convertire questo file'.format(round(time.time() - start_2), 2))
        print('Non sono stato in grado di leggere {} files su {}'.format(bad_files, len(all_files)))
        print("L'intero processo ci ha impiegato {} secondi".format(round(time.time() - start), 2))
        df = pd.DataFrame.from_records(all_files, columns=['path', 'files', 'category'])
        df['files'] = df['files'].str.replace("\n", " ")
        if os.name == 'nt': #if Windows
            desktop = os.path.join(os.path.join(os.environ['USERPROFILE']), 'Desktop')
        else: #if Unix or Linux
            desktop = os.path.join(os.path.join(os.path.expanduser('~')), 'Desktop')

        path_to_csv = desktop + r'\rec_system_data\pdf_files.csv'
        saving.save(path_to_save = path_to_csv,  object = df)


class embedder:
    @staticmethod
    def vec_for_learning(model, tagged_docs):
        """
        * Calcola i vettori del testo ed i targets
        """
        sents = tagged_docs.values
        targets, regressors = zip(*[(doc.tags[0], model.infer_vector(doc.words)) for doc in sents]) # il model crea vettori dalle words incluse nei tags a posizione zero a livello iterativo poi zippa la categoria con i vettori
        return targets, regressors

    @staticmethod
    def vectorizer(path_to_load, model_path=False, algorithm=1, epochs=40, vector_size = 300):
        """
        - Carica il modello se già esistente e specificato
        - Effettua lo split fra train e test e chiama la funzione di pulizia del testo
        - Se il modello non è specificato, crea il modello secondo gli iperparametri specificati usando Doc2Vec
        - Se l'algoritmo = 1 --> Distributed Memory, se 0 --> Bag of words
        - Dal momento che le variabili di train hanno un indice randomico ed esclusivo (derivante dallo split iniziale), creo un dataframe dove è specificato l'esatto percorso originale
        """
        extension = 'csv'
        # read only newest file
        i = 0
        while path.exists(path_to_load + r'\pdf_files_{}'.format(i) + '.' + extension):
            i += 1
        if path.exists(path_to_load + r'\pdf_files_{}'.format(i) + '.' + extension) == False:
            new_path = path_to_load + r'\pdf_files_{}'.format(i - 1) + '.' + extension
        else:
            new_path = path_to_load + r'\pdf_files_{}'.format(i) + '.' + extension
        print('reading file {}'.format(new_path))
        df = pd.read_csv(new_path)
        df.drop('Unnamed: 0', axis=1, inplace=True)
        df['files'] = df['files'].apply(lambda x: cleaner_d2v.text_cleaning(x))
        df_new = df[df['files'].map(lambda d: len(d)) > 0]
        df_tagged = df_new.apply(
            lambda r: TaggedDocument(words=r.files,
                                     tags=[r.category]), axis=1)

        if model_path:
            print('Carico il modello ed embeddizzo il testo...')
            model = embedder.load_model(model_path)
            y, X = embedder.vec_for_learning(model, df_tagged)

        else:

            print('Inizio il training...')
            cores = multiprocessing.cpu_count()
            start = time.time()
            model = Doc2Vec(dm=algorithm, vector_size=vector_size, hs=0, min_count=2, sample=0, negative=5, workers=cores, # costruisco il modello
                            epochs=epochs)
            model.build_vocab([x for x in tqdm(df_tagged.values)]) # costruisco il vocabolario
            model.train([x for x in tqdm(df_tagged.values)], total_examples=model.corpus_count, epochs=model.epochs) # traino il modello
            print('Training finito in', time.time() - start)
            y, X = embedder.vec_for_learning(model, df_tagged) # embeddizzo il testo

        df_final = pd.DataFrame(X)
        df_final['categoria'] = y
        df_final.index = df_new.index # indicizza con il vecchio indice dove alcuni documenti sono stati rimossi
        df_final['pdf'] = ['pdf_{}'.format(x) for x in range(len(df_final))] # non essenziale ma usata "for the sake of clarity"

        path_list = [df.iloc[i, -3] for i in list(df_final.index)] # crea lista di percorsi assoluti rispetto al df originale
        df_final['path'] = path_list

        if os.name == 'nt':  # if Windows
            desktop = os.path.join(os.path.join(os.environ['USERPROFILE']), 'Desktop')
        else:  # if Unix or Linux
            desktop = os.path.join(os.path.join(os.path.expanduser('~')), 'Desktop')

        # salvataggio
        path_to_csv = desktop + r'\rec_system_data\embedded_pdf_files.csv'
        if model_path:
            print('Model not overwritten')
            new_path = saving.save(path_to_save=path_to_csv, object=df_final)
            new_path_model = model_path
        else:
            model_path = path_to_load + r'\d2v_new.model'
            new_path = saving.save(path_to_save=path_to_csv, object=df_final)
            new_path_model = saving.save(path_to_save=model_path, object=model)

        return df_final, model, new_path_model, new_path

    @staticmethod
    def load_model(model_path):
        """
        * Funzione usata per caricare il modello
        """
        return Doc2Vec.load(model_path)

    @staticmethod
    def recommender(pdf_path, path_to_load):
        """
        - Passo il percorso assoluto del pdf e quello della cartella dove sono presenti gli embeddings
        - Eseguo prodotto scalare fra il pdf specificato e tutti i pdf vettorizzati per ottenere un valore di similarità
        - Ritorno una list dei 5 pdf più simili al pdf specificato
        """
        extension = 'csv'
        # read only newest file
        i = 0
        while path.exists(path_to_load + r'\embedded_pdf_files_{}'.format(i) + '.' + extension):
            i += 1
        if path.exists(path_to_load + r'\embedded_pdf_files_{}'.format(i) + '.' + extension) == False:
            new_path = path_to_load + r'\embedded_pdf_files_{}'.format(i - 1) + '.' + extension
        else:
            new_path = path_to_load + r'\embedded_pdf_files_{}'.format(i) + '.' + extension
        print('reading file {}'.format(new_path))

        df = pd.read_csv(new_path)
        df.drop('Unnamed: 0', axis=1, inplace=True)

        pdf_name = [x.split('\\')[-1].split('.pdf')[0] for x in df.path]
        df['pdf_name'] = pdf_name

        pdf = pdf_path.split('\\')[-1].split('.pdf')[0]

        def get_pdf_index(pdf_name, pdf_matrix):
            '''
            * Ritorna indice del pdf passandogli il nome
            '''
            index = pdf_matrix.index[df['pdf_name'] == pdf_name].tolist()[0]
            return index

        print('Nome documento --> {}\nCategoria documento --> {}'.format(pdf, df.iloc[get_pdf_index(pdf_name=pdf, pdf_matrix=df), -4]))

        # eseguo prodotto scalare fra il pdf selezionato e tutti i pdfs
        df['dot_products'] = df.iloc[:, :-5].dot(
            pd.Series(df.iloc[get_pdf_index(pdf_name=pdf, pdf_matrix=df), :-5]))
        sort_pdf = df.sort_values('dot_products', axis=0, ascending=False) # Ordinamento decrescente
        pdf_list = []
        for i in range(1, 6): # salta il primo pdf perchè è lo stesso
            doc = sort_pdf['pdf'].iloc[i]
            doc_name = sort_pdf['pdf_name'].iloc[i]
            doc_type = sort_pdf['categoria'].iloc[i]
            dot = sort_pdf['dot_products'].iloc[i]
            path_pdf = sort_pdf['path'].iloc[i]
            if doc in pdf:
                continue;
            else:
                print(
                    'Top {0} PDF --> {1} --> categoria: {2} --> similarità prodotto scalare {3}'.format(i, doc_name, doc_type,
                                                                                               dot))
                pdf_list.append(path_pdf)
        df.drop('dot_products', axis=1, inplace=True)
        return pdf_list

    @staticmethod
    def similar_category(category, path_to_load):
        """
        - Creo tutti i possibili match le fra categorie (semplice permutazione dove k = 2)
        - Raggruppo e calcolo la media dei vettori delle categorie
        - Calcolo la similarità fra categorie col prodotto scalare
        - Ritorno tuple contenti la prima e la seconda categoria più simile a quella specificata da "category" con i corrispettivi scores
        * Implementazione futura --> Questa funzione potrebbe anche essere lanciata una sola volta dove mi vado a costruire un dataframe
        e rilanciata se più categorie vengono aggiunte
        """
        extension = 'csv'
        # leggo solo il file più nuovo
        i = 0
        while path.exists(path_to_load + r'\embedded_pdf_files_{}'.format(i) + '.' + extension):
            i += 1
        if path.exists(path_to_load + r'\embedded_pdf_files_{}'.format(i) + '.' + extension) == False:
            new_path = path_to_load + r'\embedded_pdf_files_{}'.format(i - 1) + '.' + extension
        else:
            new_path = path_to_load +r'\embedded_pdf_files_{}'.format(i) + '.' + extension
        print('reading file {}'.format(new_path))
        df = pd.read_csv(new_path)
        df.drop('Unnamed: 0', axis=1, inplace=True)

        df = df.groupby(by = ['categoria']).mean()
        df.reset_index(level=0, inplace=True)
        df_2 = df.iloc[:, 1:] # rimuovo la categoria nel calcolo del prodotto scalare
        category_matches = [] # creo i matches e i corrispettivi prodotti scalari
        dot = []
        for row in range(len(df)):
            for pdf in range(len(df)):
                category_matches.append(df.categoria[row] + '->' + df.categoria[pdf])
                dot.append(df_2.iloc[:, row].dot(df_2.iloc[:, pdf]))

        df_final = pd.DataFrame(zip(category_matches, dot), columns=['matches', 'dot_inner_product'])
        df_final = df_final.sort_values(by=['dot_inner_product'], ascending=False)
        df_final.set_index('matches', inplace=True)
        categorie = list(df.categoria)
        similar_cat = []
        for i in categorie:
            match = df_final.loc['{}->{}'.format(category, i)]
            similar_cat.append(match)
        similar = pd.DataFrame(similar_cat)
        # calcolo una percentuale di similarità
        min = abs(similar.min().values[0])
        similar = similar + min  # per normalizzare i valori negativi sposto il minimo valore sullo 0

        max = similar.max().values[0]
        new_min = similar.min().values[0]
        similar['similarity_percentage'] = ((similar['dot_inner_product'] - new_min) / (max - new_min)) * 100  # normalizzazione
        similar = similar.sort_values(by='similarity_percentage', ascending=False)
        first_similar = similar.index[0].split('->')[1]
        second_similar = similar.index[1].split('->')[1]
        third_similar = similar.index[2].split('->')[1]
        first_score = similar.iloc[1, 1]
        second_score = similar.iloc[2, 1]
        third_score = similar.iloc[3, 1]

        print((first_similar, first_score), (second_similar, second_score), (third_similar, third_score))

        """
        N.B. Può accadere che il prodotto scalare fra due categorie uguali non riportino il valore più alto, ma questo è dovuto al fatto che 
        la media potrebbe non essere un buon indice (distribuzione non parametrica) e soprattutto che il valore di alcuni vettori
        riporti magnitudini diverse dagli altri
        """

        return (first_similar, first_score), (second_similar, second_score), (third_similar, third_score)

    @staticmethod
    def users_pre_processing(users_path):
        """
        Pre-processing del dataframe contente gli utenti
        - Riempio i NAs con "NON SPECIFICATO"
        - Transformo la variabile "sesso" a numerica binaria (1, 0)
        - Normalizzo la variabile "Età"
        - Performo il One-Hot-Encoding sulle rimanenti variabili
        """
        df = pd.read_csv(users_path, sep=';')
        sesso = {'M': 1, 'F': 0}
        df.Sesso = [sesso[item] for item in df.Sesso]
        values = {"Grado": 'NON SPECIFICATO', "Sesso": 'NON SPECIFICATO', "Categoria": 'NON SPECIFICATO',
                  "Specialita": 'NON SPECIFICATO',
                  "Qualifica": 'NON SPECIFICATO', "Ruolo": 'NON SPECIFICATO', "Sede Lavoro": 'NON SPECIFICATO',
                  "ProvinciaNascita": 'NON SPECIFICATO',
                  "Eta": 'NON SPECIFICATO'}
        df.fillna(value=values, inplace=True)
        df['Eta'] = (df['Eta'] - df['Eta'].min()) / (df['Eta'].max() - df['Eta'].min()) # normalizzo l'età
        df = pd.get_dummies(df, columns=[x for x in df.columns if x != 'Sesso' and x != 'Eta']) # performo one hot encoding su variabili categoriche ed escludo età e la variabile binaria "Sesso"
        return df

    @staticmethod
    def findksimilarusers(user_id, df, metric='cosine', k=8):
        """
        * Trovo gli utenti più simili a livello anagrafico
        """
        model_knn = NearestNeighbors(metric=metric, algorithm='auto')
        model_knn.fit(df)

        distances, indices = model_knn.kneighbors(df.iloc[user_id - 1, :].values.reshape(1, -1), n_neighbors=k + 1) # calcolo le k distanze e gli indici relativi ai k utenti simili
        similarities = 1 - distances.flatten()  # ottengo la similarità -> 1 = stesso utente
        print("{0} utenti più simili all'utente {1}:\n".format(k, user_id))
        for i in range(0, len(indices.flatten())):
            # salta lo stesso utente
            if indices.flatten()[i] + 1 == user_id:
                continue;

            else:
                print('{0}: Utente {1} : con livello di similarità {2}'.format(i, indices.flatten()[i] + 1,
                                                                       similarities.flatten()[i]))

        users = [df.iloc[x + 1, :] for x in indices] # estraggo solo gli utenti più simili e costruisco dataframe
        users = pd.DataFrame(users[0])

        return users















