def download():
    import gdown
    output = 'movielens.zip'
    url = 'https://drive.google.com/uc?id=1Manzqjg4g73ALhaVSWceaOKKKwWFShcz'
    # https://drive.google.com/file/d/1Manzqjg4g73ALhaVSWceaOKKKwWFShcz/view?usp=sharing # Clean Movielens 20m
    gdown.download(url, output, quiet=False)

    output = 'oms.zip'
    url = 'https://drive.google.com/uc?id=1EQGYw4iSDPZPUaxlNJHnt7y7t42ySCf7'
    # https://drive.google.com/file/d/1EQGYw4iSDPZPUaxlNJHnt7y7t42ySCf7/view?usp=sharing # Clean OMS Full
    gdown.download(url, output, quiet=False)


if __name__ == '__main__':
    download()
