import os
import pandas as pd
import pysam
import subprocess
import h5py
from collections import defaultdict
import argparse

# 配置参数
MINIMAP2_CMD = "minimap2 -ax splice -uf -C5 {ref} {fastq} > {sam}"
PAS_SITE_WINDOW = 100  # PAS位点窗口大小
PAU_THRESHOLD = 0.05  # polyA使用度阈值

def extract_fastq(fast5_file):
    """ 从FAST5文件提取FASTQ序列 """
    fastq_sequences = []
    try:
        with h5py.File(fast5_file, 'r') as f:
            fastq_path = '/Analyses/Basecall_1D_000/BaseCalled_template/Fastq'
            if fastq_path in f:
                fastq_data = f[fastq_path][()]
                fastq_sequence = fastq_data.decode('utf-8')
                fastq_sequences.append(fastq_sequence)
            else:
                print(f"FastQ path not found in the file: {fast5_file}")
    except Exception as e:
        print(f"Error reading Fast5 file: {e}")
    return fastq_sequences

def map_reads(fastq_file, output_sam, reference_genome):
    """ 使用minimap2比对 reads """
    cmd = MINIMAP2_CMD.format(ref=reference_genome, fastq=fastq_file, sam=output_sam)
    subprocess.run(cmd, shell=True, check=True)

def identify_pas_sites(sam_file):
    """ 识别并统计polyA位点(PAS) """
    pas_counts = defaultdict(int)
    gene_read_counts = defaultdict(int)
    
    with pysam.AlignmentFile(sam_file, "r") as sam:
        for read in sam.fetch():
            if read.is_unmapped:
                continue
            cleavage_site = read.reference_end
            pas_counts[(read.reference_name, cleavage_site)] += 1
            gene_read_counts[read.reference_name] += 1
    
    # 计算PAU
    pau_values = {site: count / gene_read_counts[site[0]] for site, count in pas_counts.items() if gene_read_counts[site[0]] > 0}
    
    # 过滤低PAU位点
    significant_pas = {site: pau for site, pau in pau_values.items() if pau > PAU_THRESHOLD}
    
    return significant_pas

def main(input_csv, fast5_dir, output_file, reference_genome):
    """ 处理ONT数据并识别PAS """
    df = pd.read_csv(input_csv)
    valid_reads = df[df["polya_is_valid"] == True]
    
    output_sam_files = []
    for _, row in valid_reads.iterrows():
        fast5_file = os.path.join(fast5_dir, row["file_path"])
        fastq_sequences = extract_fastq(fast5_file)
        
        if not fastq_sequences:
            continue
        
        fastq_file = fast5_file.replace(".fast5", ".fastq")
        with open(fastq_file, "w") as fq:
            fq.write("\n".join(fastq_sequences))
        
        sam_file = fastq_file.replace(".fastq", ".sam")
        map_reads(fastq_file, sam_file, reference_genome)
        output_sam_files.append(sam_file)
    
    # 解析比对结果
    pas_sites = {}
    for sam_file in output_sam_files:
        pas_sites.update(identify_pas_sites(sam_file))
    
    # 输出PAS结果
    with open(output_file, "w") as out:
        for (chrom, pos), pau in pas_sites.items():
            out.write(f"{chrom}\t{pos}\t{pau}\n")
    
    print("PAS identification completed. Results saved to:", output_file)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ONT PAS Site Identification")
    parser.add_argument("--input_csv", required=True, help="Path to input CSV file")
    parser.add_argument("--fast5_dir", required=True, help="Path to directory containing FAST5 files")
    parser.add_argument("--output_file", required=True, help="Path to output PAS results file")
    parser.add_argument("--reference_genome", required=True, help="Path to reference genome")
    args = parser.parse_args()
    
    main(args.input_csv, args.fast5_dir, args.output_file, args.reference_genome)
