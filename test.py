from llama_index import download_loader
import faiss
import pandas as pd

FaissReader = pd.read_json('./BigData.json')

id_to_text_map = {
    "id1": "text blob 1",
    "id2": "text blob 2",
}
index = faiss.IndexFlatL2(68)
# add embeddings to the index
index.add(...)

# initialize reader
reader = FaissReader(index)
# To load data from the Faiss index, you must specify:
# k: top nearest neighbors
# query: a 2D embedding representation of your queries (rows are queries)
k = 4
query1 = np.array([...])
query2 = np.array([...])
query=np.array([query1, query2])
documents = reader.load_data(query=query, id_to_text_map=id_to_text_map, k=k)