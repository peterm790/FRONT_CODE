import glob


def get_files():
    models = glob.glob("/terra/data/cmip5/global/historical/*")
    avail={}
    for model in models:
        uas = glob.glob(str(model)+"/r1i1p1/3hr/native/uas*")
        vas = glob.glob(str(model)+"/r1i1p1/3hr/native/vas*")
        try:
            test = uas[0]
            avail[model.split('/')[-1]] = {'uas':uas,'vas':vas}
        except:
             pass
    return avail
