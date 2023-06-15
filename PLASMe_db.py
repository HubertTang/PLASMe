import argparse
import hashlib
import os
import pathlib
import subprocess
import requests
import urllib.request
import sys
import re
import shutil


def plasmedb_cmd():
    parser = argparse.ArgumentParser(description="ARGUMENTS")
    
    parser.add_argument(
        "--keep_zip",
        default=False,
        type=bool,
        required=False,
        help="Keep the compressed database. (Default: False)")
    
    parser.add_argument(
        "--threads",
        default=8,
        type=int,
        required=False,
        help="The number of threads used to build the database (Default: 8)")
    
    plasmedb_args = parser.parse_args()

    return plasmedb_args


def connect(host='https://www.google.com/'):
    try:
        urllib.request.urlopen(host)
        return True
    except:
        return False


def check_md5(file_path):
    """Check md5 of the target file.
    """
    h = hashlib.md5()
    with open(file_path,'rb') as f: 
        while chunk := f.read(1000000*h.block_size): 
            h.update(chunk)
    return h.hexdigest()


class ProgressBar(object):
    """ https://zhuanlan.zhihu.com/p/145568973
    """
    DEFAULT = 'Progress: %(bar)s %(percent)3d%%'
    FULL = '%(bar)s %(current)d/%(total)d (%(percent)3d%%)'

    def __init__(self, total, width=40, fmt=DEFAULT, symbol='=',
                 output=sys.stderr):
        assert len(symbol) == 1

        self.total = total
        self.width = width
        self.symbol = symbol
        self.output = output
        self.fmt = re.sub(r'(?P<name>%\(.+?\))d',
            r'\g<name>%dd' % len(str(total)), fmt)

        self.current = 0

    def __call__(self):
        percent = self.current / float(self.total)
        size = int(self.width * percent)
        bar = '[' + self.symbol * size + ' ' * (self.width - size) + ']'

        args = {
            'total': self.total,
            'bar': bar,
            'current': self.current,
            'percent': percent * 100
        }
        print('\r' + self.fmt % args, file=self.output, end='')

    def done(self):
        self.current = self.total
        self()
        print('', file=self.output)


def load_curl(curl_path):
    """Load the json from the cURL file.
    """
    url_link = f""
    header_json = {}

    with open(curl_path) as curl_p:
        for l in curl_p:
            l = l.split("' \\")[0]
            if l[: 4] == 'curl':
                url_link = l.split("curl '")[1]
            elif l[: 4] == '  -H':
                json_k = l.split("  -H '")[1].split(': ', 1)[0]
                json_v = l.split(': ', 1)[1]
                header_json[json_k] = json_v

    # print(url_link, header_json)
    return url_link, header_json


def download_db(curl_link, headers, out_path):
    """Download the database using curl.
    """
    response = requests.get(curl_link, headers=headers, stream=True)
    with open(out_path, "wb") as f:

        filesize = response.headers["Content-Length"]
        chunk_size = 128 * 10000
        times = int(filesize) // chunk_size
        print("Downloading DB.zip ... ...")

        progress = ProgressBar(times, fmt=ProgressBar.FULL)
        for chunk in response.iter_content(chunk_size):
            f.write(chunk)
            progress.current += 1
            progress()
        progress.done()


def plasme_db(keep_zip=False, num_threads=8):
    """Download and build the database.
    """
    plasme_full_path = pathlib.Path(__file__).parent.resolve()
    curl_link, headers = load_curl(curl_path="db_curl")
    
    # Check if the Internet is connected.
    if not connect():
        print(f"No internet connection.")
        exit()

    # Check if the `DB` exists
    if os.path.exists(f"{plasme_full_path}/DB"):
        print(f"The target database folder ({plasme_full_path}/DB) already exists.")
        exit()

    # Check if the DB.zip exists. 
    # If so, check md5.
    # If not, download.
    db_zip_path = f"{plasme_full_path}/DB.zip"
    db_md5 = "b32263ad4fb20c04044700170b1609ef"
    if os.path.exists(db_zip_path):
        print(f"Verifying md5 ... ")
        f_md5 = check_md5(file_path=db_zip_path)
        if f_md5 != db_md5:
            print(f"DB.zip is incomplete or corrupted, redownload DB.zip ... ")
            download_db(curl_link, headers, out_path=db_zip_path)
            print(f"Verifying md5 ... ")
            f_md5 = check_md5(file_path=db_zip_path)
            if f_md5 != db_md5:
                print(f"DB.zip is incomplete or corrupted, please rerun the script to redownload the database.")
    else:
        print(f"Downloading DB.zip ... ")
        download_db(curl_link, headers, out_path=db_zip_path)
        print(f"Verifying md5 ... ")
        f_md5 = check_md5(file_path=db_zip_path)
        if f_md5 != db_md5:
            print(f"DB.zip is incomplete or corrupted, please rerun the script to redownload the database.")

    # Uncompress DB.zip and build the database
    print("Unzip the reference plasmid database ... ...")
    shutil.unpack_archive(f"{plasme_full_path}/DB.zip", plasme_full_path)

    db_dir = f"{plasme_full_path}/DB"
    print("Unzip the reference sequences ... ...")
    shutil.unpack_archive(f"{db_dir}/plsdb.zip", db_dir)
    os.remove(f"{db_dir}/plsdb.zip")

    print("Build DIAMOND and BLAST database ... ...")
    subprocess.run(f"diamond makedb --in {db_dir}/plsdb_Mar30.fna.aa -d {db_dir}/plsdb_Mar30 -p {num_threads}", shell=True)
    subprocess.run(f"makeblastdb -in {db_dir}/plsdb_Mar30.fna -dbtype nucl -out {db_dir}/plsdb_Mar30", shell=True)

    os.remove(f"{db_dir}/plsdb_Mar30.fna")
    os.remove(f"{db_dir}/plsdb_Mar30.fna.aa")

    # delete the compressed database or not
    if not keep_zip:
        os.remove(db_zip_path)


if __name__ == "__main__":

    plasmedb_args = plasmedb_cmd()

    plasme_db(keep_zip=plasmedb_args.keep_zip, 
              num_threads=plasmedb_args.threads)
