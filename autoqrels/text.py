def get_text(dataset, ids, text_field):
    if isinstance(ids, str):
        return get_text(dataset, [ids], text_field)[0]
    ids = list(ids)
    entities = dataset.lookup(ids)
    if text_field is None:
        texts = [entities[i][1] for i in ids]
    elif isinstance(text_field, str):
        texts = [getattr(entities[i], text_field) for i in ids]
    else:
        texts = ['\n'.join([getattr(entities[i], f) for f in text_field]) for i in ids]
    return texts

def query_text(dataset, query_ids, text_field):
    return get_text(dataset.queries, query_ids, text_field)

def doc_text(dataset, doc_ids, text_field):
    return get_text(dataset.docs, doc_ids, text_field)
