import argparse
from src.utils.file_handlers import inference, file_process

def main(args):
    mode = args.mode
    if mode == 'single':
        print("menggunakan mode single")
        epitope = args.epitope
        print("peptida yang digunakan {}".format(epitope))
        hla= args.hla
        print("HLA yang digunakan {}".format(hla))
        score = inference(epitope,hla)
        print(score)
    elif mode == 'multiple':
        print("menggunakan mode multiple")
        intFile = args.intdir
        print("input file adalah {}".format(intFile))
        outFolder = args.outdir
        print("output akan berada di {}".format(outFolder))
        file_process(intFile,outFolder)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='DeepImmuno-CNN command line')
    parser.add_argument('--mode',type=str,default='single',help='single mode or multiple mode')
    parser.add_argument('--epitope',type=str,default=None,help='if single mode, specifying your epitope')
    parser.add_argument('--hla',type=str,default=None,help='if single mode, specifying your HLA allele')
    parser.add_argument('--intdir',type=str,default=None,help='if multiple mode, specifying the path to your input file')
    parser.add_argument('--outdir',type=str,default=None,help='if multiple mode, specifying the path to your output folder')
    args = parser.parse_args()
    main(args)
