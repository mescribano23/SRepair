import os
import json
import re

def get_code_from_dataset(bug):
    with open('defects4j-sf.json','r') as f:
        data = json.load(f)
    buggy_code = data[bug]['fix']
    with open('code.java','w') as f:
        f.write(buggy_code)

def replace_blank(text):
    return text.replace(' ','').replace('\n','').replace('\t','')


def calculate_overlap_code(whole_text, sub_texts):
    whole_text = replace_blank(whole_text)
    covered_ranges = []

    for sub_text in sub_texts:
        sub_text = replace_blank(sub_text)

        for match in re.finditer(re.escape(sub_text), whole_text):

            covered_ranges.append((match.start(), match.end()))

    covered_ranges.sort()  
    merged_ranges = [0]*len(whole_text)


    for start, end in covered_ranges:
        for i in range(start, end):
            merged_ranges[i] = 1

    overlaped_words_num = sum(merged_ranges)
    total_words_num = len(whole_text)

    covered_code_chars = []
    for i in range(len(merged_ranges)):
        if merged_ranges[i] == 1:
            covered_code_chars.append(whole_text[i])
    covered_code_substring = ''.join(covered_code_chars)

    coverage_ratio = overlaped_words_num / total_words_num

    return coverage_ratio, covered_code_substring


def get_distribution(result_path, modified_patch_path):
    with open(result_path, 'r') as f:
        data = json.load(f)
    with open(modified_patch_path, 'r') as f:
        bug_modified_lines = json.load(f)


    overlap_bugs = []
    non_overlap_bugs = []
    for bug in data:
        bug_data = data[bug]
        covered_string = calculate_overlap_code(bug_data['input'], bug_data['query_result'])[1]

        is_modified_code_overlap = True
        for line in bug_modified_lines[bug]:
            line = replace_blank(line)
            if line not in covered_string:
                is_modified_code_overlap = False
                break
        if is_modified_code_overlap:

            overlap_bugs.append(bug)
        else:

            non_overlap_bugs.append(bug)
        data[bug]['is_modified_code_overlap'] = is_modified_code_overlap

    # with open('growingbug_pls_query_results2.json', 'w') as f:
    #     json.dump(data, f, indent=4)


