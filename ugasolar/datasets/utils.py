from pathlib import Path


def download(data_dir, *urls):
    '''Downloads data at the urls into the data_dir.

    Args:
        data_dir: The directory into which the files are downloaded.
        *urls: A list of URLs to download.

    Returns:
        A list of paths to the downloaded files.
    '''
    import requests

    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    files = []

    for url in urls:
        filename = url.rpartition('/')[2]
        filename = filename.partition('#')[0]
        filename = data_dir / filename
        files.append(filename)

        resp = requests.get(url, stream=True)
        with filename.open('wb') as f:
            for chunk in resp.iter_content(None):
                f.write(chunk)

    return files


def gunzip(*filenames, unlink=False):
    '''Decompress gzipped files.

    Args:
        *filenames: List of files to decompress.
        unlink: If True, delete the original files.

    Returns:
        A list of paths to the decompressed files.
    '''
    import gzip

    files = []

    for filename in filenames:
        filename = Path(filename)

        if filename.suffix == '.gz':
            dest = filename.with_suffix('')
        elif filename.suffix == '.tgz':
            dest = filename.with_suffix('.tar')
        else:
            msg = f"expected file path with extension '.gz' or '.tgz', got '{filename}'"
            raise ValueError(msg)

        files.append(dest)
        with dest.open('wb') as d:
            with gzip.open(filename, 'rb') as f:
                d.write(f.read())
        if unlink:
            filename.unlink()

    return files


def apply_transform(x, transform):
    '''Apply a transform to some input.

    The transform can be a callable, a list of transforms, or None.
    If the transform is callable, it is applied directly to the input.
    If the transform is a list, each member is applied in order.
    If the transform is None, the input is returned without transformation.

    Args:
        x: The input being transformed.
        transform: The transform(s) to apply.

    Returns:
        The result of transforming `x`.
    '''
    if transform is None:
        return x

    elif callable(transform):
        return transform(x)

    else:
        for t in transform:
            x = apply_transform(x, t)
        return x
