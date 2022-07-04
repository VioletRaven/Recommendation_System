'''
main
'''
import pandas as pd
from Recommendation_System.pre_processing_doc2vec import cleaner_d2v, embedder
import argparse

'''
Parser per i parametri
'''
def parser():
    ap = argparse.ArgumentParser(description = 'Parser per sistema di raccomandazione')
    ap.add_argument('-ep', '--end_path', default = None, type = str,
                    help = 'Percorso della cartella dove inserisco i nuovi files csv')
    ap.add_argument('-p', '--path', default = None, type = str,
                    help = 'Percorso della cartella contenente le cartelle dei pdf ed i pdf stessi')
    ap.add_argument('-emb', '--embedding', default = None, type = str,
                    help = 'Specificare se eseguire calcolo embedd'
                           'izzazione')
    ap.add_argument('-mp', '--model_path', default = None, type = str,
                    help = "Percorso del modello\nSe il modello è specificato l'allenamento del modello non parte")
    ap.add_argument('-pdf', default = None, type = str,
                    help = "Passa il path assoluto del pdf")
    ap.add_argument('-u', '--user', default=None, type=int,
                    help="Passa il numero riga dell'utente")
    ap.add_argument('-up', '--users_path', default=None, type=str,
                    help="Passa il percorso del file utenti")
    ap.add_argument('-c', '--category', default=None, type=str,
                    help="Passa il numero riga dell'utente")

    arg1 = ap.parse_args()

    path = arg1.path
    end_path = arg1.end_path
    emb = arg1.embedding
    model_path = arg1.model_path
    pdf = arg1.pdf
    user_id = arg1.user
    users_path = arg1.users_path
    category = arg1.category

    return path, end_path, emb, model_path, pdf, user_id, users_path, category

path, end_path, emb, model_path, pdf, user_id, users_path, category = parser()

if __name__ == '__main__':
    '''Leggo i dati dalle cartelle'''
    if path != None:
        cleaner_d2v.pdf_reader(path)

    '''Controllo se voglio eseguire il calcolo degli embedding'''
    if emb != None:
        '''Creo embeddings'''
        if end_path != None:
            if model_path != None:
                embedding, _, _, path_to_csv = embedder.vectorizer(path_to_load = end_path, model_path= model_path)
            else:
                embedding, _, _, path_to_csv = embedder.vectorizer(path_to_load = end_path)

    '''Raccomando il pdf'''
    if pdf != None:
        pdf_list = embedder.recommender(pdf_path = pdf, path_to_load = end_path)

    '''Raccomando top 8 utenti simili'''
    df = pd.DataFrame()
    if users_path != None:
        df = embedder.users_pre_processing(users_path)
    if not df.empty:
        users = embedder.findksimilarusers(user_id = user_id, df = df)
    else:
        print('Percorso utenti non specificato')

    '''Raccomando le 2 categorie più simili alla categoria utente per evitare il cold start'''


    '''control categories suggestion'''
    if category != None:
        embedder.similar_category(category = category, path_to_load = end_path)
    ''' ci servirà poi sapere che tipo di categoria viene aggiunta all'utente da poter così fare inferiezioni sulla migliore categoria affine da suggerire all'utente'''

# path = r'C:\Users\DITTA\Desktop\rec_system_pdf'
# category = 'Eventi AM' #per esempio
# pdf = r"C:\Users\DITTA\Desktop\rec_system_pdf\Area Giuridico-Legale\SMA-UCAG-001 (linee guida per la trattazione degli atti parlamentari di sindacato ispettivo e delle iniziative legislative).pdf"
# users_path = r"C:\Users\DITTA\Desktop\rec_system_data\Personale1.csv"
# user_id = 3000
# end_path = r'C:\Users\DITTA\Desktop\rec_system_data'
# returns top 199 similar users
# users = embedder.KNNsimilarusers(user_id = 13778, df = df)
# model_path = r'C:\Users\DITTA\Desktop\rec_system_data\d2v_new_9.model'
