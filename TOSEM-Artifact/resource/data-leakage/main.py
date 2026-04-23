import json
import os.path
import time
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import utils

def query_text(dataset_path):
    # 初始化 WebDriver
    result_dir = 'leakage_bugs_d4j_fix_function'
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)
    driver = webdriver.Edge()

    with open(dataset_path, 'r') as f:
        dataset = json.load(f)
    url = 'https://stack-v2.dataportraits.org/'
    driver.get(url)
    time.sleep(3)

    input_box = WebDriverWait(driver, 10).until(
        EC.presence_of_element_located((By.XPATH, '//*[@id="query-box"]'))
    )

    for bug in dataset:
        if os.path.exists(f'{result_dir}/{bug}.json'):
            continue

        text = dataset[bug]["fix"]
        if not text or len(text) == 0:
            continue

        text = text.replace('\t', '    ').strip()
        if len(text) > 2000:
            text = text[:2000]

        curr_query_data = {}
        query_result = []

        try:
            input_box.clear()
            input_box.send_keys(text) 
            time.sleep(1)
 
            WebDriverWait(driver, 10).until(
                EC.presence_of_element_located((By.XPATH, '//*[@id="card-results"]'))
            )

            results_container = driver.find_element(By.XPATH, '//*[@id="card-results"]')

            card_elements = results_container.find_elements(By.XPATH, './/div[contains(@class, "card m-3")]')

            coverage_ratio = 0
            if card_elements and len(card_elements) > 0:
                for card in card_elements:
                    card_body = card.find_element(By.XPATH, './/div[contains(@class, "card-body") and contains(@class, "text-center")]')
                    
                    result_text = driver.execute_script("""
                        return arguments[0].innerText || arguments[0].textContent;
                    """, card_body)
                    query_result.append(result_text)

                coverage_ratio, _ = utils.calculate_overlap_code(text, query_result)

            curr_query_data['coverage_ratio'] = coverage_ratio
            curr_query_data['input'] = text
            curr_query_data['query_result'] = query_result


            with open(f'{result_dir}/{bug}.json', 'w') as f:
                json.dump(curr_query_data, f, indent=4)

        except Exception as e:
            driver.quit()
            driver = webdriver.Edge()
            time.sleep(3)
            driver.get(url)
            time.sleep(3)

            input_box = WebDriverWait(driver, 10).until(
                EC.presence_of_element_located((By.XPATH, '//*[@id="query-box"]'))
            )
            print(f'Error for bug {bug}: {e}')

    driver.quit()  
    all_results = {}
    for file in os.listdir(result_dir):
        with open(f'{result_dir}/{file}', 'r') as f:
            data = json.load(f)
            all_results[file.split('.')[0]] = data
    ## sort all_results by coverage_ratio
    all_results = dict(sorted(all_results.items(), key=lambda x: x[1]['coverage_ratio'], reverse=True))

    with open('all_results.json', 'w') as f:
        json.dump(all_results, f, indent=4)


if __name__ == "__main__":
    query_text('dataset/defects4j-sf.json')
    utils.get_distribution('all_results.json','dataset/d4j_sf_patch_modified_line.json')
