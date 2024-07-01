import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt#
import numpy as np
from cycler import cycler
import matplotlib
from scipy import stats
from numpy import std, mean, sqrt
import argparse
import tldextract
import time
import os
from file_paths import *
import re
if not os.path.exists('./fig'):
    os.makedirs('fig')

#Helper Functions

def is_first_party_domains(row):
    
    row['is_first_party_domain'] = False
    if row['cookiesetter_domains'] is None:
        row['is_first_party_domain'] = True
    elif row['publisher'] in row['cookiesetter_domains']:
        row['is_first_party_domain'] = True
    return row

def domain_to_cmp_mapping(row):
    #espn, cookielaw, onetrust maps to onetrust
    '''
    #cdn.cookielaw.org maps to onetrust (Previous work)
    #https://cdn.espn.com/onetrust/otCCPAiab.js maps to onetrust (Manual Verification)
    '''
    try:
        if any([x in row['source'] for x in ['cdn.cookielaw.org', 'cdn.espn.com/onetrust/otCCPAiab.js']]):     
            row['source_domain'] = 'onetrust'
        #Quantcast (Previous work)
        elif any([x in row['source'] for x in ['quantcast.mgr.consensu.org','cmp.quantcast.com']]):
            row['source_domain'] = 'quantcast'
        elif any([x in row['source'] for x in ['consent.trustarc.com']]):
            row['source_domain'] = 'trustarc'     
        elif any([x in row['source'] for x in ['privacy-mgmt.com','sp-prod.com']]):
            row['source_domain'] = 'sourcepoint'
    except:
        pass
    
    return row

def get_cookie_setter_domains(row):
    cookie_string  = row
    domain_pattern = r'domain=([^;]+)'
    try:
        match = re.search(domain_pattern, cookie_string)
        domain = match.group(1)
        return domain
    except:
        pass
        
def extract_domain(url):
    parsed_url = tldextract.extract(url)
    domain =parsed_url.domain
    return domain

def is_third_party(domain, publisher_url):
    publisher_domain = extract_domain(publisher_url)
    if domain == publisher_domain:
        return False
    else:
        return True

def cohen_d(x,y):

    """
    #Calculate Effect Size (https://www.ncbi.nlm.nih.gov/pmc/articles/PMC3444174/) Using Cohen's method
    #Good Article about why effect size also matters https://towardsdatascience.com/effect-size-d132b0cc8669 , https://machinelearningmastery.com/effect-size-measures-in-python/, we are using pooled https://www.statisticshowto.com/pooled-standard-deviation/
    #correct if the population S.D. is expected to be equal for the two groups. (https://stackoverflow.com/questions/21532471/how-to-calculate-cohens-d-in-python), we are using pooled std https://www.statisticshowto.com/pooled-standard-deviation/
    """



    nx = len(x)
    ny = len(y)
    dof = nx + ny - 2
    return (mean(x) - mean(y)) / sqrt(((nx-1)*std(x, ddof=1) ** 2 + (ny-1)*std(y, ddof=1) ** 2) / dof)
    
def get_pairs_fig_7(df_initiator_receiver, df_initiator_receiver_usp):
     #Count all Pairs 
    df_all_same_pairs_count = df_initiator_receiver.query('initiator_domain == domain').groupby(['initiator_domain','domain'])['chainid'].nunique().reset_index(name='Chain Count').sort_values(ascending=False,by = 'Chain Count').rename(columns ={'initiator_domain':'Initiator','domain':'Receiver'})
    df_all_different_pairs_count = df_initiator_receiver.query('initiator_domain != domain').groupby(['initiator_domain','domain'])['chainid'].nunique().reset_index(name='Chain Count').sort_values(ascending=False,by = 'Chain Count').rename(columns ={'initiator_domain':'Initiator','domain':'Receiver'})
    df_all_pairs = pd.concat([df_all_same_pairs_count,df_all_different_pairs_count],ignore_index =True, sort=False)
    df_all_pairs = df_all_pairs.rename(columns={'Chain Count':'All Chain Count'})



    #Count USP Pairs 
    df_same_pairs_count_usp = df_initiator_receiver_usp.query('initiator_domain == domain').groupby(['initiator_domain','domain'])['chainid'].nunique().reset_index(name='Chain Count').sort_values(ascending=False,by = 'Chain Count').rename(columns ={'initiator_domain':'Initiator','domain':'Receiver'})
    df_different_pairs_count_usp= df_initiator_receiver_usp.query('initiator_domain != domain').groupby(['initiator_domain','domain'])['chainid'].nunique().reset_index(name='Chain Count').sort_values(ascending=False,by = 'Chain Count').rename(columns ={'initiator_domain':'Initiator','domain':'Receiver'})
    df_all_pairs_usp = pd.concat([df_same_pairs_count_usp,df_different_pairs_count_usp],ignore_index = True,sort=False)
    df_all_pairs_usp = df_all_pairs_usp.rename(columns={'Chain Count':'Consent Chain Count'})




    #Pair Analysis
    n_same_pair_chain_count_usp = df_same_pairs_count_usp['Chain Count'].sum()
    n_different_pair_chain_count_usp = df_different_pairs_count_usp['Chain Count'].sum()
    percent_pairs_with_usp = len(df_all_pairs_usp) *100/len(df_all_pairs)
    percent_no_usp = 100 - percent_pairs_with_usp





    print(f'Total number of same USP pairs in 821 websites have ({n_same_pair_chain_count_usp} chains ) {len(df_same_pairs_count_usp)} pairs')
    print(f'Total Number of Different USP Pairs in 821 websites ({n_different_pair_chain_count_usp} chains) {len(df_different_pairs_count_usp)} pairs')
    print(f'Total Number of All Uniq Pairs that appear on 821 websites {len(df_all_pairs)}')
    print(f'Total Number of All Uniq Pairs that have usp on 821 websites {len(df_all_pairs_usp)}')
    print(f'Percent of Pairs that have usp of all pairs {percent_pairs_with_usp}%')
    print(f'Percent of Pairs that have no usp of all pairs {percent_no_usp}')


    #For plotting cdf, consider only those chains that've at least 1 Chain with USP URL.
    df_merged_all_pairs = pd.merge(df_all_pairs, df_all_pairs_usp, on = ['Initiator','Receiver'], how ='inner')        
    df_merged_all_pairs['percent'] = df_merged_all_pairs['Consent Chain Count']*100/df_merged_all_pairs['All Chain Count']
    return df_merged_all_pairs

##########################################################################
##Section 4.1

def fig_1_uspapi_ranks():
    start_time = time.time()

    #plot Settings
    FIG_HALF_WIDTH = (5, 3)
    matplotlib.rcParams["pdf.fonttype"] = 42
    matplotlib.rcParams["ps.fonttype"] = 42
    matplotlib.rcParams['axes.axisbelow'] = True
    colors = [ "#5790fc","#964a8b", "#7a21dd","#e42536"]
    #Input 
    fp_uspapi = CA_DEFAULT_USP_API_PATH
    fp_website_ranks = TRANCO_RANKS_PATH
    fp_ad_chains_parsed = CA_DEFAULT_HTTP_REQUESTS_CHAINS_PATH

    #Output
    fp_CA_default_api_rank = FIG_PATH + "CA_default_api_rank_ad_resources.pdf"
    #Get Websites where we detect USP API and Merge with Ranks
    df_usprivacy_uspapi = pd.read_csv(fp_uspapi, sep = '\t')
    df_usprivacy_uspapi = df_usprivacy_uspapi[df_usprivacy_uspapi['us_api_value'].notnull()].drop_duplicates(subset = 'publisher')

    df_website_ranks = pd.read_csv(fp_website_ranks, header = None).rename(columns = {0:'rank',1:'publisher'}).head(10000)
    df_usprivacywebsites_ranks = pd.merge(df_website_ranks,df_usprivacy_uspapi, on =['publisher'], how = 'inner')

    #Get Ad-Chains and Merge with USP API
    df_ad_chains_parsed = pd.read_csv(fp_ad_chains_parsed, sep = '\t')
    df_websites_with_ad_resources = df_ad_chains_parsed.drop_duplicates(subset = ['website']).rename(columns = {'website':'publisher'})


    df_websites_with_ad_resources_ranks = pd.merge(df_website_ranks, df_websites_with_ad_resources, how = 'inner', on = 'publisher')[['publisher','rank']]




    bins = 20
    increment = 10000 / bins
    current_min = 0
    current_max = increment 
    uspapi_columns = []
    ad_resources_columns = []
    for i in range(bins):
        in_range_uspapi_column = [c for c in df_usprivacywebsites_ranks['rank'] if (c >=current_min and c < current_max)]
        in_range_ad_resources_columns = [c for c in df_websites_with_ad_resources_ranks['rank'] if (c >=current_min and c < current_max)] 
        current_min += increment
        current_max += increment
        uspapi_columns.append(100*len(in_range_uspapi_column)/500)
        ad_resources_columns.append(100*len(in_range_ad_resources_columns)/500)




    positions = (0,1000,2000,3000,4000,5000,6000,7000,8000,9000,10000)
    labels = ("0","1k", "2k","3k","4k","5k","6k","7k","8k","9k","10k")

    fig = plt.figure(figsize=FIG_HALF_WIDTH)
    ax = fig.add_subplot(111)

    ax.tick_params(bottom=False, left=False, right=False)
    plt.tight_layout()
    plt.grid(linestyle=':')

    x = np.arange(0,10000,500)
    ax.step(x,ad_resources_columns, color = colors[0], label="Has A&A",where="post")
    ax.step(x,uspapi_columns,color = colors[1], label="Has USP API",where="post")

    ax.set_xlim(0,10000)
    ax.set_xticks(positions)
    ax.set_xticklabels(labels)
    ax.set_ylabel('Percentage of Websites')
    ax.set_ylabel('Tranco Rank')
    ax.legend()

    for spine in ('top', 'right', 'bottom', 'left'):
        plt.gca().spines[spine].set_visible(False)
    ax.yaxis.grid(True)
    ax.xaxis.grid(False)
    ax.set_axisbelow(True)
    plt.savefig(fp_CA_default_api_rank, bbox_inches='tight')
    plt.show()
    print(f"Function took {(time.time() - start_time)/60} minutes")
    "Function took 1.1088217496871948 minutes"


def fig_2_uspapi_category():
    start_time = time.time()
    #Input 
    fp_uspapi = CA_DEFAULT_USP_API_PATH
    #Get Categories 
    fp_website_category = FORTIGUARD_CATEGORY_PATH

    #Output
    fp_usprivacy_categories = FIG_PATH + "CA_default_uspapi_categories.pdf"

    #Load Data
    df_usprivacy_uspapi = pd.read_csv(fp_uspapi, sep = '\t')
    df_usprivacy_uspapi = df_usprivacy_uspapi[df_usprivacy_uspapi['us_api_value'].notnull()].drop_duplicates(subset='publisher')
    df_websites_category = pd.read_csv(fp_website_category, sep ='\t')


    n_percent_categories = 100*df_websites_category.publisher.nunique()/10000
    print(f'Total Number of websites among Top 10k where we get categories from the Fortiguard {df_websites_category.publisher.nunique()}({n_percent_categories}%)')
    print(f'Total Number of categories detected among top 10k {df_websites_category.category.nunique()}')
    print(f'Total Number of websites where we detect the presence of USP API {df_usprivacy_uspapi.publisher.nunique()}')
    #Get Web Categories for Pages where USP API is present
    df_website_uspapi_categories = pd.merge(df_websites_category[['publisher','category']],df_usprivacy_uspapi, on = 'publisher', how = 'inner')
    n_percent_website_upapi_categories = 100*df_website_uspapi_categories.publisher.nunique()/df_usprivacy_uspapi.publisher.nunique()
    print(f'We detected USP API among {df_usprivacy_uspapi.publisher.nunique()}, we were able to successfully get category among {df_website_uspapi_categories.publisher.nunique()} {n_percent_website_upapi_categories}% websites')

    #Filter Out Websites without USP API and get their categories
    df_website_no_uspapi_category = df_websites_category[~df_websites_category.publisher.isin(df_usprivacy_uspapi.publisher)]

    #Get Top 10 Categories where we detect max number of USP API
    df_category_counts_usprivacy = df_website_uspapi_categories.groupby('category').size().sort_values(ascending=False).reset_index().rename (columns= {0:'usprivacy_categories'}).head(10)
    df_category_counts_nonusprivacy = df_website_no_uspapi_category.groupby('category').size().sort_values(ascending=False).reset_index().rename (columns= {0:'no_usprivacy_categories'})

    #Plot Settings
    FIG_HALF_WIDTH = (5, 3)
    # Six colors
    matplotlib.rcParams['axes.prop_cycle'] = cycler('color', ["#5790fc",  "#e42536", "#92dadd",  "#a96b59", "#7a21dd"])

    #Font
    matplotlib.rcParams["pdf.fonttype"] = 42
    matplotlib.rcParams["ps.fonttype"] = 42
    matplotlib.rcParams['axes.axisbelow'] = True

    '''
    x-axis is sorted by categories where we detect the most number of USP API. We show top 10 categories. Each bar shows the Percentage of Websites in that category where we detect USP API and where we do not detect USP API.
    '''
    fig, axs = plt.subplots(1,1,figsize=FIG_HALF_WIDTH)
    df_plot_data = pd.merge(df_category_counts_usprivacy, df_category_counts_nonusprivacy, on = 'category', how = 'inner') 
    df_plot_data['usprivacy_categories_percent'] = df_plot_data['usprivacy_categories']*100/10000
    df_plot_data['no_usprivacy_categories_percent'] = df_plot_data['no_usprivacy_categories']*100/10000
    #Sort categories based on websites where we detect most USP API 
    df_plot_data = df_plot_data.sort_values(by='usprivacy_categories_percent', ascending = False)
    percent_categories_top10_coverage = (100*df_plot_data.usprivacy_categories.sum()/df_usprivacy_uspapi.publisher.nunique())
    print(f'Focus on Top 10 Categories that cover {df_plot_data.usprivacy_categories.sum()}({percent_categories_top10_coverage}%)')
    #get required columns
    df_plot_data_req =df_plot_data[['category', 'usprivacy_categories_percent','no_usprivacy_categories_percent']]
    columns_to_rename = {
            "usprivacy_categories_percent": "USP API Detected",
            "no_usprivacy_categories_percent": "USP API Not Detected",
        }
    df_plot_data_req = df_plot_data_req.rename(columns=columns_to_rename)
    df_plot_data_req = df_plot_data_req.set_index('category')
    axs.tick_params(which="both", bottom=False, left=False, right=False)
    plt.grid(linestyle=':')
    plt.tight_layout() 
    for spine in ('top', 'right', 'bottom', 'left'):
        plt.gca().spines[spine].set_visible(False)

    df_plot_data_req.plot.bar(stacked= True, ax =axs,  rot=90)
    axs.legend( bbox_to_anchor=(1.00, 1), loc='upper right', fontsize = 8)
    plt.ylabel("Percentage of Websites ", fontsize=11)
    plt.xlabel("", fontsize=0)
    plt.xticks(fontsize=9)
    plt.yticks(fontsize=9)

    axs.yaxis.grid(True)
    axs.xaxis.grid(False)
    axs.set_axisbelow(True)
    plt.savefig(fp_usprivacy_categories, bbox_inches='tight')
    print(f"Function took {(time.time() - start_time)/60} minutes") 

    """ 
    Total Number of websites among Top 10k where we get categories from the Fortiguard 9897(98.97%)
    Total Number of categories detected among top 10k 75
    Total Number of websites where we detect the presence of USP API 821
    We detected USP API among 821, we were able to successfully get category among 821 100.0% websites
    Focus on Top 10 Categories that cover 646(78.68453105968331%)
    """


def tab_2_cookie_count():
    start_time = time.time()
    #Input
    fp_usprivacy_cookies = CA_DEFAULT_USPRIVACY_COOKIES_PATH
    #Load Data
    df_usprivacy_cookies = pd.read_csv(fp_usprivacy_cookies)

    df_usprivacy_cookies = df_usprivacy_cookies[df_usprivacy_cookies['value'].notnull()]
    #Output 
    fp_cookie_table = FIG_PATH + "noformat_usprivacy_default_cookie_count.tex"

    

    df_usprivacy_cookies_website_count = df_usprivacy_cookies.groupby(['name'])['publisher'].nunique().reset_index().rename(columns = {"name":"Name","publisher":"defaultWebsites"}).sort_values(by =['defaultWebsites'],ascending = False)
    #Save Table for Default Only 
    df_usprivacy_cookies_website_count.to_latex(fp_cookie_table, index = False)
    print(f"Function took {(time.time() - start_time)/60} minutes") 
    "Function took 0.00039194027582804364 minutes"

def generate_uspapi_adoption_rate_different_ccpa_criteria_numbers():
        start_time = time.time()
        fp_CA_default_april2023 = CA_DEFAULT_USP_API_PATH
        fp_CA_default_adchains_april2023 = CA_DEFAULT_HTTP_REQUESTS_CHAINS_PATH     
        fp_website_ranks = TRANCO_RANKS_PATH
        

        df_CA_default_uspapi_april2023_top10k = pd.read_csv(fp_CA_default_april2023, sep = '\t')
        df_CA_default_adchains_april2023 = pd.read_csv(fp_CA_default_adchains_april2023, sep = '\t')

        #Get Top 10k websites
        df_website_ranks = pd.read_csv(fp_website_ranks, header = None).rename(columns = {0:'rank',1:'publisher'}).head(10000)


        #Filter Out only those websites where we detect the USP API among Top 10k
        df_CA_default_uspapi_april2023_top10k = df_CA_default_uspapi_april2023_top10k[df_CA_default_uspapi_april2023_top10k['us_api_value'].notnull()] 


        #Consdering only top 5k websites, filter out those websites that have USP API
        df_websites_ranks_top5k = df_website_ranks[df_website_ranks['rank'].between(0, 5000, inclusive="both")]
        df_CA_default_uspapi_april2023_top5k = pd.merge(df_CA_default_uspapi_april2023_top10k, df_websites_ranks_top5k,on=['publisher'], how = 'inner')

        TOP_10k = 10000
        TOP_5k = 5000
        #Filter websites that have A&A among top10k and among top 5k
        TOP_10k_ANA_WEBSITES_ONLY = df_CA_default_adchains_april2023.website.nunique()
        TOP_5k_ANA_WEBSITES_ONLY = pd.merge(df_CA_default_adchains_april2023,df_websites_ranks_top5k,left_on=['website'],right_on=['publisher'], how = 'inner')['website'].nunique()



        #USP API Adoption Rate among top 10k websites
        percent_CA_default_uspapi_april2023_top10k_adoption_rate_top10k = df_CA_default_uspapi_april2023_top10k.publisher.nunique()*100/TOP_10k

        #USP API Adoption Rate among top 5k websites
        percent_CA_default_uspapi_april2023_top10k_adoption_rate_top5k = df_CA_default_uspapi_april2023_top5k.publisher.nunique()*100/TOP_5k

        #USP API Adoption Rate among top 10k websites that have A&A resources.
        percent_CA_default_uspapi_april_2023_adoption_rate_top10k_ANA =  df_CA_default_uspapi_april2023_top10k.publisher.nunique()*100/TOP_10k_ANA_WEBSITES_ONLY 

        #USP API Adoption Rate among top 5k websites that have A&A resources.
        percent_CA_default_uspapi_april_2023_adoption_rate_top5k_ANA = df_CA_default_uspapi_april2023_top5k.publisher.nunique()*100/TOP_5k_ANA_WEBSITES_ONLY

        print(f'USP API Adoption Rate considering all top 10 website {df_CA_default_uspapi_april2023_top10k.publisher.nunique()}({percent_CA_default_uspapi_april2023_top10k_adoption_rate_top10k}%)')

        print(f'USP API Adoption Rate considering all top 5k websites {df_CA_default_uspapi_april2023_top5k.publisher.nunique()}({percent_CA_default_uspapi_april2023_top10k_adoption_rate_top5k}%)')


        print(f'Considering top 10k websites, {TOP_10k_ANA_WEBSITES_ONLY} websites have A&A resources. Then USP API adoption rate is {df_CA_default_uspapi_april2023_top10k.publisher.nunique()}({percent_CA_default_uspapi_april_2023_adoption_rate_top10k_ANA}%)')         

        print(f'Considering top 5k websites,  {TOP_5k_ANA_WEBSITES_ONLY} websites have A&A resources. Then USP API adoption rate is {df_CA_default_uspapi_april2023_top5k.publisher.nunique()}({percent_CA_default_uspapi_april_2023_adoption_rate_top5k_ANA}%)')  
        print(f"Function took {(time.time() - start_time)/60} minutes") 
        """
        USP API Adoption Rate considering all top 10 website 821(8.21%)
        USP API Adoption Rate considering all top 5k websites 438(8.76%)
        Considering top 10k websites, 6799 websites have A&A resources. Then USP API adoption rate is 821(12.075305191939991%)
        Considering top 5k websites,  3425 websites have A&A resources. Then USP API adoption rate is 438(12.788321167883211%)
        """


def generate_usprivacy_cookies_numbers():
    """
    The function carries out basic analysis of usprivacy cookies and what cookies are present
   
    """
    start_time = time.time()
    #Input
    fp_usprivacy_cookies = CA_DEFAULT_USPRIVACY_COOKIES_PATH 
    #Load Data
    df_usprivacy_cookies = pd.read_csv(fp_usprivacy_cookies)

    df_usprivacy_cookies = df_usprivacy_cookies[df_usprivacy_cookies['value'].notnull()]
 
    ################Analysis where we found more than two cookies
    df_recommend_usprivacy_cookie = df_usprivacy_cookies[df_usprivacy_cookies['name'] == 'usprivacy']
    print(f'Number of Websites where we detect the presence of recommended usprivacy cookie {df_recommend_usprivacy_cookie.publisher.nunique()}')
    print(f'Total Number of Websites where we detect the presence of usprivacy cookies {df_usprivacy_cookies.publisher.nunique()}')
    df_ncookies_per_website = df_usprivacy_cookies.groupby('publisher')['name'].nunique().reset_index().rename(columns = {"name":"count"}).sort_values(by = ['count'] , ascending=False)
    df_two_cookies_per_website = df_ncookies_per_website[df_ncookies_per_website['count'] == 2]
    print(f'We detect presence of two US Privacy Cookies on Websites {len(df_two_cookies_per_website)}')
    df_two_cookies_publisher = pd.merge(df_usprivacy_cookies,df_two_cookies_per_website['publisher'], on = 'publisher', how = 'inner')
    df_usprivacy_ntv_as_us_privacy = df_two_cookies_publisher[(df_two_cookies_publisher['name'] == 'usprivacy') | (df_two_cookies_publisher['name'] == 'ntv_as_us_privacy')].groupby(['publisher'])['name'].nunique().reset_index()
    df_usprivacy_ntv_as_us_privacy= df_usprivacy_ntv_as_us_privacy[df_usprivacy_ntv_as_us_privacy['name']>1]
    print(f'Number of websites  two US Privacy Cookies usprivacy and ntv_as_us_privacy {len(df_usprivacy_ntv_as_us_privacy)}')
    df_usprivacy_us_privacy = df_two_cookies_publisher[(df_two_cookies_publisher['name'] == 'usprivacy') | (df_two_cookies_publisher['name'] == 'us_privacy')].groupby(['publisher'])['name'].nunique().reset_index()
    df_usprivacy_us_privacy= df_usprivacy_us_privacy[df_usprivacy_us_privacy['name']>1]
    print(f'Number of websites with US Privacy Cookies usprivacy and us_privacy {len(df_usprivacy_us_privacy)}')
    print(f"Function took {(time.time() - start_time)/60} minutes") 

    """
    Number of Websites where we detect the presence of recommended usprivacy cookie 321
    Total Number of Websites where we detect the presence of usprivacy cookies 358
    We detect presence of two US Privacy Cookies on Websites 35
    Number of websites  two US Privacy Cookies usprivacy and ntv_as_us_privacy 25
    Number of websites with US Privacy Cookies usprivacy and us_privacy 10
    """



def generate_uspapi_cookie_together_numbers():
    """
    Analysis of co-occurrence of the USP API and the US PRIVACY COOKIES. 
    """
    start_time = time.time()
    #Input
    fp_usprivacy_cookies = CA_DEFAULT_USPRIVACY_COOKIES_PATH 
    fp_uspapi = CA_DEFAULT_USP_API_PATH
    df_usprivacy_cookies = pd.read_csv(fp_usprivacy_cookies)
    df_usprivacy_cookies = df_usprivacy_cookies[df_usprivacy_cookies['value'].notnull()]

    df_uspapi = pd.read_csv(fp_uspapi, sep = '\t')
    df_uspapi = df_uspapi[df_uspapi['us_api_value'].notnull()]
    
    print(f'Considering all websites and all webpages where we detected the presence of USPAPI {df_uspapi.publisher.nunique()}')
    print(f'Conisering all websites and all webpages where we detected the presence of usprivacy cookies {df_usprivacy_cookies.publisher.nunique()}')

    """
    We want to answer if both USP API and recommended usprivacy cookie is present considring only homepage
    """
    df_recommend_usprivacy_cookie = df_usprivacy_cookies[df_usprivacy_cookies['name'] == 'usprivacy']
    print(f'Conisering all websites and all webpages where we detected the presence of recommended usprivacy cookies {df_recommend_usprivacy_cookie.publisher.nunique()}')
    #Filter HomePage Only 
    df_usprivacy_cookies = df_usprivacy_cookies[df_usprivacy_cookies['ishomepage']==True]
    df_recommend_usprivacy_cookie = df_recommend_usprivacy_cookie[df_recommend_usprivacy_cookie['ishomepage']==True]
    df_uspapi = df_uspapi[df_uspapi['ishomepage']==True]

    print(f'Considering Only Home Pages where we detected the presence of USPAPI {df_uspapi.publisher.nunique()}')
    print(f'Considering Only Home Pages where we detected the presence of US Privacy Cookies {df_usprivacy_cookies.publisher.nunique()}')
    print(f'Conisdering Only Home Pages where we detected the presence of recommended US Privacy Cookie {df_recommend_usprivacy_cookie.publisher.nunique()}')



    """
    Analysis of presence of both USP API and recommended usprivacy cookie.
    """
    #Filter Out websites where both Recommended USprivacy cookie and USPAPI is present.
    df_both = pd.merge(df_recommend_usprivacy_cookie,df_uspapi, on = ['publisher','ishomepage','page_url'], how = 'inner')
    df_both['present'] = 'both'
    #Filter Out websites where US Privacy Cookies is only Present
    df_websites_with_usprivacy_cookies_only = df_recommend_usprivacy_cookie.merge(df_both['publisher'], on=['publisher'],how='left', indicator=True)
    df_websites_with_usprivacy_cookies_only = df_websites_with_usprivacy_cookies_only[df_websites_with_usprivacy_cookies_only['_merge']=='left_only'].drop_duplicates(subset = ['publisher'])
    df_websites_with_usprivacy_cookies_only['present'] = 'cookie_only'
    print(f'Number of Websites with recommended usprivacy cookie only {df_websites_with_usprivacy_cookies_only.publisher.nunique()}')
    df_websites_with_uspapi_only = df_uspapi.merge(df_both['publisher'], on = ['publisher'], how = 'left', indicator=True) 
    df_websites_with_uspapi_only = df_websites_with_uspapi_only[df_websites_with_uspapi_only['_merge']=='left_only']
    df_websites_with_uspapi_only['present'] ='api_only' 
    print(f'Number of Websites where USPAPI is present only {df_websites_with_uspapi_only.publisher.nunique()}')
    print(f'Number of Websites where USPAPI and recommended usprivacy both are Present {df_both.publisher.nunique()}')
    print(f'Total Number of Websites where we detected the presence of both recommended US Privacy Cookie and API {df_both.publisher.nunique()+df_websites_with_uspapi_only.publisher.nunique()+df_websites_with_usprivacy_cookies_only.publisher.nunique()}')
    
    """
    Considering all websites and all webpages where we detected the presence of USPAPI 821
    Conisering all websites and all webpages where we detected the presence of usprivacy cookies 358
    Conisering all websites and all webpages where we detected the presence of recommended usprivacy cookies 321
    Considering Only Home Pages where we detected the presence of USPAPI 705
    Considering Only Home Pages where we detected the presence of US Privacy Cookies 313
    Conisdering Only Home Pages where we detected the presence of recommended US Privacy Cookie 282
    Number of Websites with recommended usprivacy cookie only 15
    Number of Websites where USPAPI is present only 438
    Number of Websites where USPAPI and recommended usprivacy both are Present 267
    Total Number of Websites where we detected the presence of both recommended US Privacy Cookie and API 720
    Function took 0.0032371004422505696 minutes
    """

    print(f"Function took {(time.time() - start_time)/60} minutes") 


############################################################################
#Section 4.2

def fig_3_uspapi_writing():

    start_time = time.time()
    fp_uspapi_read_write = CA_1YYN_READ_WRITE_API_PATH
    fp_uspapi_default = CA_DEFAULT_USP_API_PATH
    df_uspapi_read_write = pd.read_csv(fp_uspapi_read_write, sep = '\t') 
    df_uspapi_default = pd.read_csv(fp_uspapi_default, sep = '\t')
    df_uspapi_default = df_uspapi_default[df_uspapi_default['us_api_value'].notnull()].drop_duplicates(subset = 'publisher')

    df_api_write = df_uspapi_read_write[df_uspapi_read_write['api_write_stack_found'] == True]
    nwebsites = df_api_write.website.nunique()
    print(f'Total Number of Website detected where API is being written {nwebsites}')
    n_api_write_websites_overlap_with_default = pd.merge(df_api_write, df_uspapi_default, left_on ='website',right_on='publisher', how = 'inner').publisher.nunique()
    print(f'Out of {nwebsites}, {n_api_write_websites_overlap_with_default} have the USP API in our default crawl from California')
    
    #We try to map Top (CMPs) domains that were calling USP API to their respective companies. Some are mentioned in the previous work : https://dl.acm.org/doi/pdf/10.1145/3419394.3423647
    #OneTrust cdn.cookielaw.org
    #Quantcast quantcast.mgr.consensu.org 
    #Source Point is sp-prod.net (https://ccpa.sp-prod.net/ccpa.js)
    #Source Point is privacy.mgmt (https://cdn.privacy-mgmt.com/ccpa.js)

    df_api_write.loc[df_api_write['api_write_stack_domain']=="cookielaw", 'api_write_stack_domain'] = 'onetrust'
    df_api_write.loc[df_api_write['api_write_stack_domain']=="consensu", 'api_write_stack_domain'] = 'quantcast'
    df_api_write.loc[df_api_write['api_write_stack_domain']=="privacy-mgmt", 'api_write_stack_domain'] = 'sourcepoint'
    df_api_write.loc[df_api_write['api_write_stack_domain']=="sp-prod", 'api_write_stack_domain'] = 'sourcepoint'
    
    #Focus only on third parties trying to write USP API to generate figure.
    df_api_write['is_third_party'] = df_api_write.apply(lambda x : is_third_party(x['api_write_stack_domain'],x['website']), axis =1 )
    df_api_write = df_api_write[df_api_write['is_third_party']==True]
    #Plot Dataframe
    df_sources = df_api_write.groupby('api_write_stack_domain')['website'].nunique().reset_index(name='total').rename(columns = {'api_write_stack_domain':'sources'}).sort_values(by =['total'], ascending = False)
    
    top_sources = df_sources.nlargest(10,'total')
    print(top_sources)
    FIG_THIRD_WIDTH = (5, 2.4)

    matplotlib.rcParams["pdf.fonttype"] = 42
    matplotlib.rcParams["ps.fonttype"] = 42
    matplotlib.rcParams['axes.axisbelow'] = True

    plt.rcParams['axes.prop_cycle'] = cycler('color', ["#5790fc", "#f89c20", "#e42536", "#964a8b", "#9c9ca1", "#7a21dd"])
    fig, ax = plt.subplots(1,1,figsize=FIG_THIRD_WIDTH)
    ax.tick_params(which="both", bottom=False, left=False, right=False)
    plt.grid(linestyle=':')
    plt.tight_layout() 
    for spine in ('top', 'right', 'bottom', 'left'):
        plt.gca().spines[spine].set_visible(False)

    sns.barplot(top_sources, x="sources",y="total")
    plt.xticks(rotation=90,fontsize=10)
    plt.xlabel('')
    plt.ylabel('Number of Websites', fontsize = 10)
    plt.title('')
    #overlap    
    ax.yaxis.grid(True)
    ax.xaxis.grid(False)
    plt.savefig(FIG_PATH + "CA_tainted_api_write.pdf", bbox_inches='tight')
    print(f"Function took {(time.time() - start_time)/60} minutes") 

    """
    Total Number of Website detected where API is being written 234
    Out of 234, 183 have the USP API in our default crawl from California
    Websites where more than two parties trying to write USP API 8
                 sources  total
    8             inmobi     86
    17       sourcepoint     60
    13    privacy-center     17
    20        uniconsent      8
    9            iubenda      4
    16            sndimg      3
    5   consentframework      2
    14    privacymanager      2
    11          nocookie      2
    19         transcend      1
    Function took 0.02337107261021932 minutes
    """





def fig_4_uspapi_reading():
    start_time = time.time()
    FIG_THIRD_WIDTH = (5, 2.4)

    matplotlib.rcParams["pdf.fonttype"] = 42
    matplotlib.rcParams["ps.fonttype"] = 42
    matplotlib.rcParams['axes.axisbelow'] = True

    plt.rcParams['axes.prop_cycle'] = cycler('color', ["#5790fc", "#f89c20", "#e42536", "#964a8b", "#9c9ca1", "#7a21dd"])

    fp_uspapi_read_write = CA_1YYN_READ_WRITE_API_PATH
    df_uspapi_read_write = pd.read_csv(fp_uspapi_read_write, sep = '\t') 
    df_api_read = df_uspapi_read_write[df_uspapi_read_write['api_read_stack_found'] == True]

    #Consider Only Third Parties Reading USP API
    df_api_read['is_third_party'] = df_api_read.apply(lambda x : is_third_party(x['api_read_stack_domain'],x['website']), axis =1 )
    df_api_read = df_api_read[df_api_read['is_third_party']==True]

    nwebsites = df_api_read.website.nunique()
    print(f'Total Number of Website detected where API is being Read {nwebsites}')
    df_api_read = df_api_read.groupby('api_read_stack_domain')['website'].nunique().reset_index().rename(columns = {'api_read_stack_domain':'sources', 'website': 'napireadwebsites'}).sort_values(by =['napireadwebsites'], ascending = False)
    top_sources = df_api_read.nlargest(10,'napireadwebsites')
    print(top_sources)



    FIG_THIRD_WIDTH = (5, 2.4)

    matplotlib.rcParams["pdf.fonttype"] = 42
    matplotlib.rcParams["ps.fonttype"] = 42
    matplotlib.rcParams['axes.axisbelow'] = True

    plt.rcParams['axes.prop_cycle'] = cycler('color', ["#5790fc", "#f89c20", "#e42536", "#964a8b", "#9c9ca1", "#7a21dd"])
    fig, ax = plt.subplots(1,1,figsize=FIG_THIRD_WIDTH)
    ax.tick_params(which="both", bottom=False, left=False, right=False)
    plt.grid(linestyle=':')
    plt.tight_layout() 
    for spine in ('top', 'right', 'bottom', 'left'):
        plt.gca().spines[spine].set_visible(False)

    sns.barplot(top_sources, x="sources",y="napireadwebsites")
    plt.xticks(rotation=90,fontsize = 10)
    plt.xlabel('')
    plt.ylabel('Number of Websites',fontsize = 10)
    plt.title('')
    #overlap
    ax.yaxis.grid(True)
    ax.xaxis.grid(False)
    plt.savefig(FIG_PATH + "CA_tainted_api_read.pdf", bbox_inches='tight')
    print(f"Function took {(time.time() - start_time)/60} minutes") 


    """
   Total Number of Website detected where API is being Read 3590
               sources  napireadwebsites
    115   googletagmanager              2218
    89         doubleclick              1275
    80              criteo               829
    202           pubmatic               782
    130            indexww               524
    1             33across               456
    125           id5-sync               441
    113         googleapis               416
    208         quantserve               408
    114  googlesyndication               392
    Function took 0.13661741415659587 minutes
    """



def fig_5_uspapi_consistency_read():
    start_time = time.time()
    #LOAD DATA
    fp_uspapi_read_write = CA_1YYN_READ_WRITE_API_PATH
    fp_taintedapi_resources = CA_1YYN_RESOURCES_PATH
    df_ana_domains = pd.read_csv(EASY_PRIVACY_LIST_DOMAINS_PATH, sep ='\t')

    ana_list = df_ana_domains['domain'].tolist()

    df_uspapi_read_write = pd.read_csv(fp_uspapi_read_write, sep = '\t') 
    df_taintedapi_resources_parsed = pd.read_csv(fp_taintedapi_resources, sep = '\t')

    #extract domain from website_domain
    df_uspapi_read_write['website_domain'] = df_uspapi_read_write['website'].apply(extract_domain)

    #Filter Javascript HTTP Requests that results in the load of Javascript and keep only A&A Requests
    df_taintedapi_resources_parsed_js = df_taintedapi_resources_parsed[df_taintedapi_resources_parsed['type']=='script']
    df_taintedapi_resources_parsed_js = df_taintedapi_resources_parsed_js[
        (df_taintedapi_resources_parsed_js['domain'].isin(ana_list))
    ]
    #Consider Only A&A JS reading
    df_api_read = df_uspapi_read_write[df_uspapi_read_write['api_read_stack_found'] == True]
    df_api_read = df_api_read[
        (df_api_read['api_read_stack_domain'].isin(ana_list))
    ]



    """
    This contains consistency analysis: How frequently A&A third parties read USP API whenever they loaded successfully.
    """ 
    #Count A&A Third Parties Javascript across websites.
    df_thirdparties_js_count = df_taintedapi_resources_parsed_js.groupby(['domain'])['website'].nunique().reset_index(name='nwebsitecountthirdparty')

    #Count A&A Third Parties Javascript Reading USP API across websites
    df_api_read = df_api_read.groupby('api_read_stack_domain')['website'].nunique().reset_index().rename(columns = {'api_read_stack_domain':'sources', 'website': 'napireadwebsites'}).sort_values(by =['napireadwebsites'], ascending = False)

 

    #Consider Only those A&A Third Parties Javascript that were present on at least one website and read the USP API
    df_thirdparties_api_read_merged = pd.merge(df_thirdparties_js_count,df_api_read,left_on = ['domain'], right_on=['sources'],how = 'inner') 

    #Percentage of times A&A Third Parties Javascript read USP API whenever they loaded successfully. We consider only those A&A JS that read USP API on at least one website. 
    df_thirdparties_api_read_merged['percent_read'] = 100 * df_thirdparties_api_read_merged['napireadwebsites']/df_thirdparties_api_read_merged['nwebsitecountthirdparty']

    #We calculate of all the A&A Javascript that loaded on website, what percentage of them reads USP API
    # We use df_thirdparties_api_read_merged and not df_api_read 
    percent_reading_USP_API = 100*len(df_thirdparties_api_read_merged)/len(df_thirdparties_js_count)

    print(f'We detect the presence of total of {len(df_thirdparties_js_count)} A&A Javascript resources across 10k websites.')
    print(f' We found  {len(df_thirdparties_api_read_merged)} ({percent_reading_USP_API}%)A&A resources reading USP ')
    print(f'Not Reading script {100-percent_reading_USP_API}')

    ##########Very few parties appear more than 100 times. Only two times they appear twice 
    df_thirdparties_api_read_merged = df_thirdparties_api_read_merged[df_thirdparties_api_read_merged['percent_read']<=100]
    n_read_percent = 100* len(df_thirdparties_api_read_merged[df_thirdparties_api_read_merged['percent_read']==100])/len(df_thirdparties_api_read_merged)
    print(f'We find that {n_read_percent} of these A&A domains read the USP API consistently, whenever they load')
    #PLOTING CDF
    FIG_THIRD_WIDTH = (5, 2.4)
    matplotlib.rcParams["pdf.fonttype"] = 42
    matplotlib.rcParams["ps.fonttype"] = 42
    matplotlib.rcParams['axes.axisbelow'] = True
    plt.rcParams['axes.prop_cycle'] = cycler('color', ["#5790fc", "#f89c20", "#e42536", "#964a8b", "#9c9ca1", "#7a21dd"])

    fig, ax = plt.subplots(1,1,figsize=FIG_THIRD_WIDTH)
    plt.grid(linestyle=':')

    ax.tick_params(which="both", bottom=False, left=False, right=False)
    for spine in ('top', 'right', 'bottom', 'left'):
        plt.gca().spines[spine].set_visible(False)

    sns.ecdfplot(data=df_thirdparties_api_read_merged, x="percent_read", ax = ax, stat='percent')
    
    ax.set_ylabel('CDF')
    ax.set_xlabel('Percentage of USP API Reads by A&A JavaScript')
    ax.yaxis.grid(True)
    ax.xaxis.grid(False)
    ax.set_axisbelow(True)
    plt.tight_layout()
    plt.savefig(FIG_PATH + "CA_tainted_api_read_cdf.pdf",bbox_inches='tight')
    """
    Generating Figure 5
    We detect the presence of total of 2631 A&A Javascript resources across 10k websites.
    We found  257 (9.76814899277841%)A&A resources reading USP 
    Not Reading script 90.2318510072216
    We find that 48.627450980392155 of these A&A domains read the USP API consistently, whenever they load
    Function took 0.6709295630455017 minutes
    """

    print(f"Function took {(time.time() - start_time)/60} minutes") 


def fig_6_usprivacycookie_writing():
    start_time = time.time()
    fp_us_privacy_set_get_cookie = CA_DEFAULT_US_PRIVACY_COOKIE_SET_PATH 
    df_cookie_set_get = pd.read_csv(fp_us_privacy_set_get_cookie, sep ='\t')
    # Filter first party usprivacy cookies
    df_cookie_set =  df_cookie_set_get[df_cookie_set_get['description']=='Document.cookie setter']
    df_cookie_set = df_cookie_set[df_cookie_set['is_consent_signal'] == True] 
    df_cookie_set['cookiesetter_domains'] = df_cookie_set.arguments.apply(get_cookie_setter_domains)
    df_cookie_set = df_cookie_set.apply(is_first_party_domains, axis = 1)
    #Filter First Party Cookies Only
    df_cookie_set =df_cookie_set[df_cookie_set['is_first_party_domain']==True]
    df_cookie_set = df_cookie_set.apply(domain_to_cmp_mapping, axis = 1)

    df_cookie_set_usprivacy = df_cookie_set[df_cookie_set['consent_signal_name']=='usprivacy']
    print(f'Total Number of Websites where US Privacy Cookie is being Set {df_cookie_set.publisher.nunique()}')
    print(f'Total Number of Websites where recommended usprivacy cookie is being Set {df_cookie_set_usprivacy.publisher.nunique()}')


    #Calculate how many publishers have more than two sources trying to set up recommend usprivacy cookie on same webpage 
    df_publishers_usprivacy_set_by_two = df_cookie_set_usprivacy.groupby(['page_url'])['source_domain'].nunique().sort_values(ascending = False).reset_index(name='count')
    df_publishers_usprivacy_set_by_two = df_publishers_usprivacy_set_by_two[df_publishers_usprivacy_set_by_two['count']>1]
    df_publishers_merged_usprivacy = pd.merge(df_publishers_usprivacy_set_by_two,df_cookie_set, how ='inner', on = 'page_url')
    df_publishers_merged_usprivacy = df_publishers_merged_usprivacy[df_publishers_merged_usprivacy['consent_signal_name']=='usprivacy']
    print(f'We observe that on {df_publishers_merged_usprivacy.publisher.nunique()} websites on the same page there were at least two parties trying to set usprivacy cookie')
    
    
    #For Plotting consider only third parties trying to setup usprivacy cookies
    df_cookie_set['is_third_party'] = df_cookie_set.apply(lambda x : is_third_party(x['source_domain'],x['publisher']), axis =1 )
    df_cookie_set = df_cookie_set[df_cookie_set['is_third_party']==True]

    #Count Presence of Third or First Parties Sources across websites
    top_sources = df_cookie_set.groupby(['source_domain'])['publisher'].nunique().reset_index().rename(columns ={'publisher':'nwebsites'}).sort_values(ascending = False, by = 'nwebsites').head(10)
    print(top_sources)

    FIG_THIRD_WIDTH = (5, 2.4)
    matplotlib.rcParams["pdf.fonttype"] = 42
    matplotlib.rcParams["ps.fonttype"] = 42
    matplotlib.rcParams['axes.axisbelow'] = True

    plt.rcParams['axes.prop_cycle'] = cycler('color', ["#5790fc", "#f89c20", "#e42536", "#964a8b", "#9c9ca1", "#7a21dd"])
    fig, ax = plt.subplots(1,1,figsize=FIG_THIRD_WIDTH)
    
    ax.tick_params(which="both", bottom=False, left=False, right=False)
    plt.grid(linestyle=':')
    plt.tight_layout() 

    for spine in ('top', 'right', 'bottom', 'left'):
        plt.gca().spines[spine].set_visible(False)
    sns.barplot(top_sources, x="source_domain",y="nwebsites")

    plt.xticks(rotation=90, fontsize = 10)
    plt.xlabel("")
    plt.ylabel('Number of Websites', fontsize = 10)
    ax.yaxis.grid(True)
    ax.xaxis.grid(False)
    plt.savefig(FIG_PATH + "CA_default_usprivacy_cookiesetter.pdf", bbox_inches='tight')
    plt.show()
    print(f"Function took {(time.time() - start_time)/60} minutes") 

    """
    Total Number of Websites where US Privacy Cookie is being Set 370
    Total Number of Websites where recommended usprivacy cookie is being Set 335
    We observe that on 5 websites on the same page there were at least two parties trying to set usprivacy cookie
    source_domain  nwebsites
    29      onetrust        173
    28           ntv         52
    16     futurecdn         21
    2       adthrive         15
    33     quantcast         15
    19    intergient         13
    23       ketchjs          8
    38     snigelweb          7
    40      trustarc          6
    30         osano          6
    Function took 0.7054258346557617 minutes
    """





def tab_3_param_consent_table():
    start_time = time.time()
    fp_CA_tainted_parsed_ad_chains = CA_1YYN_HTTP_REQUESTS_CHAINS_PATH
    fp_CA_default_parsed_ad_chains = CA_DEFAULT_HTTP_REQUESTS_CHAINS_PATH

    fp_tainted_consent_signals_table = FIG_PATH + "tainted_consent_signals_notformat.tex"
    #Load Data
    df_CA_tainted_parsed_ad_chains = pd.read_csv(fp_CA_tainted_parsed_ad_chains, sep = '\t')
    df_CA_default_parsed_ad_chains = pd.read_csv(fp_CA_default_parsed_ad_chains, sep = '\t') 

    df_CA_tainted_consent_signal = df_CA_tainted_parsed_ad_chains[df_CA_tainted_parsed_ad_chains['is_consent_signal_present']==True]
    df_CA_default_consent_signal = df_CA_default_parsed_ad_chains[df_CA_default_parsed_ad_chains['is_consent_signal_present']==True]

    df_CA_default_consent_signal = df_CA_default_consent_signal[df_CA_default_consent_signal['consent_signal_value'].notnull()]
    df_CA_tainted_consent_signal = df_CA_tainted_consent_signal[df_CA_tainted_consent_signal['consent_signal_value'].notnull()]
    print(f'In Default, Unique Websites where we observe consent signals {df_CA_default_consent_signal.website.nunique()}') 
    print(f'In CA Tainted, Unique Website, where we observe consent signals {df_CA_tainted_consent_signal.website.nunique()}')   
    
    df_CA_tainted_consent_signal = df_CA_tainted_consent_signal.groupby(['consent_signal_name'])['website'].nunique().reset_index(name = 'Tainted Websites')
    df_CA_tainted_consent_signal = df_CA_tainted_consent_signal.rename(columns = {'consent_signal_name':'Parameter'})
    df_CA_tainted_consent_signal = df_CA_tainted_consent_signal.sort_values(by=['Tainted Websites'], ascending = False) 
    df_CA_tainted_consent_signal = df_CA_tainted_consent_signal.head(15)
    df_CA_tainted_consent_signal[['Parameter','Tainted Websites']].to_latex(fp_tainted_consent_signals_table, index = False)
    print(f"Function took {(time.time() - start_time)/60} minutes") 


    """
    In Default, Unique Websites where we observe consent signals 1033
    In CA Tainted, Unique Website, where we observe consent signals 3433
    Function took 2.2895411968231203 minutes 
    """


###################################
#Section 4.3 

def generate_taint_crawl_validation_numbers():

    start_time = time.time()
    fp_CA_tainted_https_requests = CA_1YYN_HTTP_REQUESTS_CHAINS_PATH
    fp_CA_tainted_api = CA_1YYN_USP_API_PATH

    df_CA_tainted_parsed_https_requests = pd.read_csv(fp_CA_tainted_https_requests, sep = '\t')
    df_CA_tainted_uspapi = pd.read_csv(fp_CA_tainted_api, sep = '\t')


    """
    We want to investigate of all the websites and webpages, what percentage of the USP string returned from USP API contains our tainted value?
    """
    df_CA_tainted_uspapi_cleaned = df_CA_tainted_uspapi[df_CA_tainted_uspapi['us_api_value'].notnull()]
    df_CA_tainted_value_detected = df_CA_tainted_uspapi_cleaned[df_CA_tainted_uspapi_cleaned['us_api_value']=='1YYN']

    print(f'Total number of unique websites crawled {df_CA_tainted_uspapi.publisher.nunique()}')
    print(f'Total number of unique pages crawled {df_CA_tainted_uspapi.page_url.nunique()}')

    print(f'Total number of websites where we successfully extracted uspapi value {df_CA_tainted_uspapi_cleaned.publisher.nunique()}')
    print(f'Total number of webpages where we successfully extracted uspapi value {df_CA_tainted_uspapi_cleaned.page_url.nunique()}')

    print(f'Total number of websites where USP API has tainted value {df_CA_tainted_value_detected.publisher.nunique()}')
    print(f'Total number of web pages where USP API has tainted value {df_CA_tainted_value_detected.page_url.nunique()}')

    """
    1) we want to investigate within CA tainted api crawl, what percentage of URLs where we detect consent signals have our tainted value

    """
    df_CA_tainted_parsed_ad_chains_consent_urls = df_CA_tainted_parsed_https_requests[df_CA_tainted_parsed_https_requests['is_consent_signal_present']==True]
    df_CA_tainted_parsed_ad_chains_consent_urls = df_CA_tainted_parsed_ad_chains_consent_urls[df_CA_tainted_parsed_ad_chains_consent_urls['consent_signal_value'].notnull()]
    print(f'Total number of URLs in taint crawl where we detect CCPA signal {len(df_CA_tainted_parsed_ad_chains_consent_urls)}')
    n_count_taint_urls= len(df_CA_tainted_parsed_ad_chains_consent_urls[df_CA_tainted_parsed_ad_chains_consent_urls['consent_signal_value'].astype(str)=='1YYN'])
    c_count_notaint_urls = len(df_CA_tainted_parsed_ad_chains_consent_urls[df_CA_tainted_parsed_ad_chains_consent_urls['consent_signal_value'].astype(str)!='1YYN'])
    print(f'Total number of urls where we detect our tainted 1YYN {n_count_taint_urls} ({(n_count_taint_urls*100)/len(df_CA_tainted_parsed_ad_chains_consent_urls)}%)')
    print(f'Total number of urls where we do not detect our tainted 1YYN {c_count_notaint_urls} ({(c_count_notaint_urls*100)/len(df_CA_tainted_parsed_ad_chains_consent_urls)}%)')
    print(f"Function took {(time.time() - start_time)/60} minutes") 

    
    """
    Total number of unique websites crawled 9992
    Total number of unique pages crawled 69739
    Total number of websites where we successfully extracted uspapi value 9982
    Total number of webpages where we successfully extracted uspapi value 69549
    Total number of websites where USP API has tainted value 9982
    Total number of web pages where USP API has tainted value 69549
    Total number of URLs in taint crawl where we detect CCPA signal 421497
    Total number of urls where we detect our tainted 1YYN 354416 (84.08505873114161%)
    Total number of urls where we do not detect our tainted 1YYN 67081 (15.914941268858378%)
    Function took 1.2036792675654093 minutes 
    """


def generate_default_chain_numbers():
    start_time = time.time()
    #Check all parameters with taint value
    fp_parsed_ad_chains = CA_DEFAULT_HTTP_REQUESTS_CHAINS_PATH

    fp_CA_default_uspapi = CA_DEFAULT_USP_API_PATH    
    df_parsed_ad_chains = pd.read_csv(fp_parsed_ad_chains, sep = '\t')
    #Get the presence of USP API on default websites
    df_CA_default_uspapi = pd.read_csv(fp_CA_default_uspapi, sep = '\t')
    df_CA_default_uspapi = df_CA_default_uspapi[df_CA_default_uspapi.us_api_value.notnull()]


    #Get Only adchains for only those intitator_receiver that combines with 821 publishers.
    df_CA_default_mergedapi_presence = pd.merge(df_parsed_ad_chains, df_CA_default_uspapi.drop_duplicates(subset=['publisher'])['publisher'], left_on = ['website'], right_on=['publisher'], how = 'inner')




    df_default_consent_urls = df_parsed_ad_chains[df_parsed_ad_chains['is_consent_signal_present']==True]
    df_default_consent_urls = df_default_consent_urls[df_default_consent_urls['consent_signal_value'].notnull()]
    print(f'Total Number of URLs in default with usp in default {len(df_default_consent_urls)}')

    df_default_consent_urls_api_presence = df_CA_default_mergedapi_presence[df_CA_default_mergedapi_presence['is_consent_signal_present']==True]
    df_default_consent_urls_api_presence = df_CA_default_mergedapi_presence[df_CA_default_mergedapi_presence['consent_signal_value'].notnull()]


    n_consent_signal_urls = len(df_default_consent_urls)
    n_total_chains_default = df_parsed_ad_chains.chainid.nunique()
    n_total_chains_default_api_presence = df_CA_default_mergedapi_presence.chainid.nunique()
    percent_chains_not_rooted_publishers_uspapi = 100*(n_total_chains_default- n_total_chains_default_api_presence)/ n_total_chains_default
    n_total_consent_chains_api_presence = df_default_consent_urls_api_presence.chainid.nunique()
    print(f'In CA default crawl, we observe {n_consent_signal_urls} HTTP requests that contain the USP String')
    print(f'Total Number of Uniq Ad-Chains in CA Default {n_total_chains_default}')
    print(f'Total Number of Ad-chains in CA Default 821 websites {n_total_chains_default_api_presence}')
    print(f'Percentage of  Chains in websites that donot adopt USP API {percent_chains_not_rooted_publishers_uspapi}%')
    print(f'Total Number of uniq Ad-chains in CA Default that contains usp url {n_total_consent_chains_api_presence}({n_total_consent_chains_api_presence*100/n_total_chains_default_api_presence}%)')


    #Number of chains that contain exactly one USP signal in 821 USP API websites 
    n_chains_with_one_usp = df_default_consent_urls_api_presence.groupby(['chainid'])['url'].size().reset_index(name='Count')
    n_chains_with_one_usp = len(n_chains_with_one_usp[n_chains_with_one_usp['Count']==1])
    percent_n_chains_one_usp = 100*n_chains_with_one_usp/n_total_consent_chains_api_presence
    print(f'Total of these chains containted exactly one {n_chains_with_one_usp} {percent_n_chains_one_usp}% out of {n_total_consent_chains_api_presence} chains')

    #To calculate Median merge with original chains first, get last element which contains pos, subtract 1 from pos (since we want to count edges but our chain node starts from 1).
    df_default_consent_urls_api_presence =  df_default_consent_urls_api_presence.drop_duplicates(subset=['chainid'])
    df_default_consent_urls_api_presence = df_default_consent_urls_api_presence[['chainid']]
    df_consent_signal_chains_merged_original = pd.merge(df_default_consent_urls_api_presence, df_CA_default_mergedapi_presence, on = 'chainid', how = 'inner')

    df_consent_signal_chains_merged_original['pos'] = df_consent_signal_chains_merged_original['pos'] - 1
    last_element_chains_with_pos = df_consent_signal_chains_merged_original.sort_values('pos', ascending=False).drop_duplicates(['chainid'])
    n_median_length_chain = last_element_chains_with_pos.pos.median()

    print(f'Median Length of Chains with USP URLs {n_median_length_chain}')

    print(f"Function took {(time.time() - start_time)/60} minutes") 


    """
    Total Number of URLs in default with usp in default 319269
    In CA default crawl, we observe 319269 HTTP requests that contain the USP String
    Total Number of Uniq Ad-Chains in CA Default 3102021
    Total Number of Ad-chains in CA Default 821 websites 1214540
    Percentage of  Chains in websites that donot adopt USP API 60.84681567275012%
    Total Number of uniq Ad-chains in CA Default that contains usp url 218541(17.993726019727635%)
    Total of these chains containted exactly one 171866 78.64245153083404% out of 218541 chains
    Median Length of Chains with USP URLs 5.0
    Function took 1.3483649134635924 minutes
    """


def tab_4_5_6_generate_initiator_receiver_tables():
    start_time = time.time()
    #Input Data
    fp_CA_default_april2023_initiator_receiver = CA_DEFAULT_HTTP_REQUESTS_CHAINS_PATH

    #Load Data
    df_CA_default_april2023_initiator_receiver = pd.read_csv(fp_CA_default_april2023_initiator_receiver, sep = '\t')


    #Output
    fp_tab_receivers_count = FIG_PATH + "tab_receivers_count_noformat.tex"
    fp_tab_initiators_count =FIG_PATH + "tab_initiators_count_noformat.tex"
    fp_top_15_usp_pairs = FIG_PATH + "tab_top_50_pairs_noformat.tex"
    """
    Generate Initiator-Receiver Table considering all chains in 10k websites in CA Default
    """
    df_default_consent_urls = df_CA_default_april2023_initiator_receiver[df_CA_default_april2023_initiator_receiver['is_consent_signal_present']==True]

    #For every initiator, count total number of unique receivers
    df_initiators_count_usp = df_default_consent_urls.groupby(['initiator_domain'])['domain'].nunique().reset_index(name ="Total Receivers").sort_values(ascending=False, by ='Total Receivers').rename(columns ={'initiator_domain':'Initiator'})
    #For every receiver, count total number of unique initiators.

    df_receivers_count_usp = df_default_consent_urls.groupby(['domain'])['initiator_domain'].nunique().reset_index(name ="Total Initiators").sort_values(ascending=False, by ='Total Initiators').rename(columns ={'domain':'Receiver'})


    df_pairs_with_usp = df_default_consent_urls.groupby(['initiator_domain','domain'])['chainid'].count().reset_index(name='Chain Count').sort_values(ascending=False,by = 'Chain Count').rename(columns ={'initiator_domain':'Initiator','domain':'Receiver'})

    """
    We want to save inititors, receivers and pairs containg USP 
    """

    df_initiators_count_usp.head(15).to_latex(fp_tab_initiators_count, index = False)
    df_receivers_count_usp.head(15).to_latex(fp_tab_receivers_count,index = False) 

    #Generate Table for top 15.
    df_pairs_with_usp.head(15).to_latex(fp_top_15_usp_pairs, index = False)
    print(f"Function took {(time.time() - start_time)/60} minutes") 
    """
    Function took 1.072621723016103 minutes
    """

def fig_7_initiator_receiver_consistency():
    start_time = time.time()
    #Input Data
    fp_CA_default_april2023_initiator_receiver = CA_DEFAULT_HTTP_REQUESTS_CHAINS_PATH
    fp_CA_default_april2023_uspapi = CA_DEFAULT_USP_API_PATH
    df_ana_domains = pd.read_csv(EASY_PRIVACY_LIST_DOMAINS_PATH, sep ='\t')
    ana_list = df_ana_domains['domain'].tolist()


    #Get the presence of USP API on default websites
    df_CA_default_april2023_uspapi = pd.read_csv(fp_CA_default_april2023_uspapi, sep = '\t')
    df_CA_default_april2023_uspapi = df_CA_default_april2023_uspapi[df_CA_default_april2023_uspapi.us_api_value.notnull()]


    #Load Data
    df_CA_default_april2023_initiator_receiver = pd.read_csv(fp_CA_default_april2023_initiator_receiver, sep = '\t')

    #Top 30 Ad-Exchanges
    df_ad_exchanges = pd.read_csv(AD_EXCHANGES_TOP_30_PATH, sep ='\t',header = None).rename(columns ={0:'exchanges'})
    df_ad_exchanges['exchange_domain'] = df_ad_exchanges['exchanges'].apply(extract_domain)
    exchanges_list = df_ad_exchanges['exchange_domain'].tolist()




    #Output
    fp_fig_cdf_pairs_usp  = FIG_PATH +"fig_fraction_pairs_usp.pdf" 

    """
    Consider all URLs chains restricted to 821 websites where we detect USP API in the default crawl.
    """


    df_CA_default_april2023_initiator_receiver_mergedapi_presence = pd.merge(df_CA_default_april2023_initiator_receiver, df_CA_default_april2023_uspapi.drop_duplicates(subset=['publisher'])['publisher'], left_on = ['website'], right_on=['publisher'], how = 'inner')

    """
    Consider Only those URLs chains that have usp and restrict to 821 websites where we detect USP API in the default crawl.

    """

    df_CA_default_april2023_receiver_consentsignals_uspapi = df_CA_default_april2023_initiator_receiver_mergedapi_presence[df_CA_default_april2023_initiator_receiver_mergedapi_presence['is_consent_signal_present']==True] 

    print(f'Total Number of Ad-Chains restricted to 821 websites where API is present (correct this) {df_CA_default_april2023_initiator_receiver_mergedapi_presence.chainid.nunique()}')
    print(f'Total Number of URLs containing USP parameter restricted to 821 websites where we detect usp api {len(df_CA_default_april2023_receiver_consentsignals_uspapi)}')

        # only consider A & A Pairs. Either Initiator or Receiver is A&A 
    df_CA_default_april2023_initiator_receiver_anapairs = df_CA_default_april2023_initiator_receiver_mergedapi_presence[
        (df_CA_default_april2023_initiator_receiver['initiator_domain'].isin(ana_list)) |
        (df_CA_default_april2023_initiator_receiver['domain'].isin(ana_list))
    ]

    df_CA_default_april2023_initiator_receiver_consentsignals_anapairs = df_CA_default_april2023_initiator_receiver_anapairs[df_CA_default_april2023_initiator_receiver_anapairs['is_consent_signal_present']==True]

    #Filter Out Those Pairs that Are Ad-Exchanges: 1) Either initiator is an ad-exchange or receiver is an ad-exchange.
    df_CA_default_april2023_initiator_receiver_mergedapi_presence_adexchanges = df_CA_default_april2023_initiator_receiver_mergedapi_presence[
        (df_CA_default_april2023_initiator_receiver_mergedapi_presence['initiator_domain'].isin(exchanges_list)) |
        (df_CA_default_april2023_initiator_receiver_mergedapi_presence['domain'].isin(exchanges_list))
    ]

    df_CA_default_april2023_receiver_consentsignals_uspapi_adexchanges = df_CA_default_april2023_initiator_receiver_mergedapi_presence_adexchanges[df_CA_default_april2023_initiator_receiver_mergedapi_presence_adexchanges['is_consent_signal_present']==True]
    ###########################################################################################################################################################
    #For Filtering Http request with tracking pixel 1) First, filter out tracking pixel and then pick the last node because last node(http request) results in the loading of a pixel. 2) Then filter out those tracking pixels that contains USP 
    
    df_CA_default_april2023_initiator_receiver_mergedapi_presence_trackingpixels = df_CA_default_april2023_initiator_receiver_mergedapi_presence[df_CA_default_april2023_initiator_receiver_mergedapi_presence['is_tracking_pixel']==True]
    df_CA_default_april2023_initiator_receiver_mergedapi_presence_trackingpixels_lastnode = df_CA_default_april2023_initiator_receiver_mergedapi_presence_trackingpixels.sort_values('pos', ascending=False).drop_duplicates(['chainid'])

    ##Filter Out tracking pixel pairs with consent signal
    df_CA_default_april2023_initiator_receiver_mergedapi_presence_trackingpixels_lastnode_consent_signals = df_CA_default_april2023_initiator_receiver_mergedapi_presence_trackingpixels_lastnode[df_CA_default_april2023_initiator_receiver_mergedapi_presence_trackingpixels_lastnode['is_consent_signal_present']==True]


    #Get a Dataframe that gets 1) A&A Pairs, 2) Ad Exchange Pairs, 3) Tracking Pixel Pair
    print('Consider all A&A Pairs')
    df_merged_ana_pairs = get_pairs_fig_7(df_CA_default_april2023_initiator_receiver_anapairs, df_CA_default_april2023_initiator_receiver_consentsignals_anapairs)

    print('Consider Ad Exchange Pairs Only pairs below')
    df_merged_adexchanges = get_pairs_fig_7(df_CA_default_april2023_initiator_receiver_mergedapi_presence_adexchanges, df_CA_default_april2023_receiver_consentsignals_uspapi_adexchanges)

    print('Consider Pairs Containing Tracking Pixels')
    df_merged_trackingpixels = get_pairs_fig_7(df_CA_default_april2023_initiator_receiver_mergedapi_presence_trackingpixels_lastnode,df_CA_default_april2023_initiator_receiver_mergedapi_presence_trackingpixels_lastnode_consent_signals)

    #Consistency Analysis.
    n_consistent_ana_pairs = 100*len(df_merged_ana_pairs[df_merged_ana_pairs['percent'] ==100])/ len(df_merged_ana_pairs)
    n_consistent_adexchange_pairs = 100*len(df_merged_adexchanges[df_merged_adexchanges['percent']==100])/len(df_merged_adexchanges)
    n_consistent_tracking_pixel_pairs = 100*len(df_merged_trackingpixels[df_merged_trackingpixels['percent']==100])/len(df_merged_trackingpixels)

    print(f'In A&A, {n_consistent_ana_pairs}% communicate consistently')
    print(f'In ad-exchange, {n_consistent_adexchange_pairs}% communicate consistently')
    print(f'In Tracking Pixel Pairs, {n_consistent_tracking_pixel_pairs}% communicate consistently')



    #PLOT CDF


    plt.rcParams["pdf.fonttype"] = 42
    plt.rcParams["ps.fonttype"] = 42
    plt.rcParams['axes.axisbelow'] = True
    plt.rcParams['axes.prop_cycle'] = cycler('color', ["#5790fc", "#f89c20", "#e42536", "#964a8b", "#9c9ca1", "#7a21dd"])
    FIG_HALF_WIDTH = (5, 3)
    FIG_THIRD_WIDTH = (4, 2.4)

    fig = plt.figure(figsize=FIG_THIRD_WIDTH)
    ax = fig.add_subplot(111)
    plt.grid(linestyle=':')

    ax.tick_params(which="both", bottom=False, left=False, right=False)
    for spine in ('top', 'right', 'bottom', 'left'):
        plt.gca().spines[spine].set_visible(False)

    sns.ecdfplot(data=df_merged_ana_pairs, x="percent", ax = ax, stat='percent', label = 'A&A Pairs')
    sns.ecdfplot(data=df_merged_adexchanges, x="percent", ax = ax, stat='percent', label = 'Ad Exchange Pairs')
    sns.ecdfplot(data=df_merged_trackingpixels, x="percent", ax = ax, stat='percent', label = 'Tracking Pixel Pairs')

    ax.legend( bbox_to_anchor=(0.50, 1.05),fontsize = 8)
    ax.set_ylabel('CDF')
    ax.set_xlabel('Percentage of Chains with USP per Pair')
    ax.yaxis.grid(True)
    ax.xaxis.grid(False)
    ax.set_axisbelow(True)
    plt.tight_layout()

    plt.savefig(fp_fig_cdf_pairs_usp)
   
    print(f"Function took {(time.time() - start_time)/60} minutes") 

    """
    Total Number of Ad-Chains restricted to 821 websites where API is present 1214540
    Total Number of URLs containing USP parameter restricted to 821 websites where we detect usp api 282320
    Consider all A&A Pairs
    Total number of same USP pairs in 821 websites have (40331 chains ) 85 pairs
    Total Number of Different USP Pairs in 821 websites (140809 chains) 3203 pairs
    Total Number of All Uniq Pairs that appear on 821 websites 33346
    Total Number of All Uniq Pairs that have usp on 821 websites 3288
    Percent of Pairs that have usp of all pairs 9.860253103820549%
    Percent of Pairs that have no usp of all pairs 90.13974689617945
    Consider Ad Exchange Pairs Only pairs below
    Total number of same USP pairs in 821 websites have (14674 chains ) 9 pairs
    Total Number of Different USP Pairs in 821 websites (81988 chains) 990 pairs
    Total Number of All Uniq Pairs that appear on 821 websites 4030
    Total Number of All Uniq Pairs that have usp on 821 websites 999
    Percent of Pairs that have usp of all pairs 24.789081885856078%
    Percent of Pairs that have no usp of all pairs 75.21091811414392
    Consider Pairs Containing Tracking Pixels
    Total number of same USP pairs in 821 websites have (1969 chains ) 19 pairs
    Total Number of Different USP Pairs in 821 websites (13462 chains) 571 pairs
    Total Number of All Uniq Pairs that appear on 821 websites 8819
    Total Number of All Uniq Pairs that have usp on 821 websites 590
    Percent of Pairs that have usp of all pairs 6.690100918471482%
    Percent of Pairs that have no usp of all pairs 93.30989908152851
    In A\&A, 47.08029197080292% communicate consistently
    In ad-exchange, 38.33833833833834% communicate consistently
    In Tracking Pixel Pairs, 58.644067796610166% communicate consistently
    Function took 1.3311857024828593 minutes
    """


def fig_8_top_receivers_params():


    start_time = time.time()
    fp_parsed_ad_chains = CA_DEFAULT_HTTP_REQUESTS_CHAINS_PATH
    fp_ana_domains = EASY_PRIVACY_LIST_DOMAINS_PATH
    df_parsed_ad_chains = pd.read_csv(fp_parsed_ad_chains, sep = '\t')
    df_ana_domains = pd.read_csv(fp_ana_domains, sep ='\t')
    ana_list = df_ana_domains['domain'].tolist()

    matplotlib.rcParams["pdf.fonttype"] = 42
    matplotlib.rcParams["ps.fonttype"] = 42    
    FIG_THIRD_WIDTH = (4, 2.4)

    df_default_consent_urls = df_parsed_ad_chains[df_parsed_ad_chains['is_consent_signal_present']==True]
    print(f'Total Number of URLs in default with usp in default {len(df_default_consent_urls)}')
    df_default_consent_urls = df_default_consent_urls[df_default_consent_urls['consent_signal_value'].notnull()]

    #Consisder only those pairs where either initiator or receiver is A&A 
    df_default_consent_urls = df_default_consent_urls[
        (df_default_consent_urls['initiator_domain'].isin(ana_list)) |
        (df_default_consent_urls['domain'].isin(ana_list))
    ]
 

    df_urls_usprivacysignals= df_default_consent_urls[df_default_consent_urls['consent_signal_name'] == 'us_privacy']
    df_urls_nonusprivacysignals = df_default_consent_urls[df_default_consent_urls['consent_signal_name'] != 'us_privacy']
    df_us_privacy_urls_domains = df_urls_usprivacysignals.groupby(['domain'])['url'].count().reset_index(name='us_privacy_url_count').rename(columns = {"index":"domain"})
    df_non_privacy_urls_domains = df_urls_nonusprivacysignals.groupby(['domain'])['url'].count().reset_index(name='non_us_privacy_url_count').rename(columns = {"index":"domain"})



    df_merged_privacyurls= pd.merge(df_us_privacy_urls_domains,df_non_privacy_urls_domains, how = 'outer', on =['domain'])
    df_merged_privacyurls=df_merged_privacyurls.fillna(0)
    df_merged_privacyurls['total'] = df_merged_privacyurls['us_privacy_url_count'] + df_merged_privacyurls['non_us_privacy_url_count']


    #Top_20_thirdparties stackbar
    df_merged_privacyurls_top = df_merged_privacyurls.sort_values(ascending=False, by=['total']).head(20)
    df_merged_privacyurls_top = df_merged_privacyurls_top[['domain','us_privacy_url_count','non_us_privacy_url_count']]

    df_merged_privacyurls_top = df_merged_privacyurls_top.set_index('domain')

    n_total = df_merged_privacyurls.total.sum()
    n_percent_us_privacy =df_merged_privacyurls.us_privacy_url_count.sum()*100/n_total
    n_percent_non_us_privacy =df_merged_privacyurls.non_us_privacy_url_count.sum()*100/n_total

    print(f'Total URLS (only A&A Pairs) with usp detected {n_total}')
    print(f'Recommended us_privacy urls (A&A pairs)  {df_merged_privacyurls.us_privacy_url_count.sum()}({n_percent_us_privacy})%')
    print(f'Other Keys us_privacy urls (A&A pairs){df_merged_privacyurls.non_us_privacy_url_count.sum()}({n_percent_non_us_privacy})%')


    matplotlib.rcParams['axes.axisbelow'] = True
    matplotlib.rcParams['axes.prop_cycle'] = cycler('color', ["#5790fc", "#f89c20", "#e42536", "#964a8b", "#9c9ca1", "#7a21dd"])

    fig, ax = plt.subplots(1,1,figsize=FIG_THIRD_WIDTH)
    for spine in ('top', 'right', 'bottom', 'left'):
        plt.gca().spines[spine].set_visible(False)
    plt.grid(linestyle=':')
    plt.tight_layout() 

    ax.tick_params(which="both", bottom=False, left=False, right=False)
    df_merged_privacyurls_top.plot.bar(stacked=True, ax = ax)
    plt.xlabel("")
    ax.legend(['us_privacy parameter','other parameter names'],bbox_to_anchor=(1.00,1), fontsize = 8)
    ax.set_ylabel('USP URLs',fontsize=9)
    ax.yaxis.grid(True)
    ax.xaxis.grid(False)
    plt.savefig(FIG_PATH + "CA_taintedapi_consent_signals_domains.pdf",bbox_inches='tight')

    print(f"Function took {(time.time() - start_time)/60} minutes") 

    """
    Total Number of URLs in default with usp in default 319269
    Total URLS (only A&A Pairs) with usp detected 290139.0
    Recommended us_privacy urls (A&A pairs)  231648.0(79.84035238282341)%
    Other Keys us_privacy urls (A&A pairs)58491.0(20.159647617176596)%
    Function took 1.0819907585779827 minutes
    """


###################################
#SECTION 4.4


def generate_default_gpc_optout_numbers():
    start_time = time.time()
    df_default_uspapi = pd.read_csv(CA_DEFAULT_USP_API_PATH, sep = '\t')
    df_gpc_uspapi = pd.read_csv(CA_GPC_USP_API_PATH, sep = '\t')

    df_default_uspapi = df_default_uspapi[df_default_uspapi['us_api_value'].notnull()]
    df_gpc_uspapi = df_gpc_uspapi[df_gpc_uspapi['us_api_value'].notnull()]


    print(f'We detected the presence of USP API in CA default on {df_default_uspapi.publisher.nunique()} websites')
    print(f'We detected the presence of USP API in CA GPC on {df_gpc_uspapi.publisher.nunique()} websites')

    #Pick one page randomly per website
    #We randomly flip rows and pick the first one
    df_default_uspapi_onepage =df_default_uspapi.sample(frac=1, random_state = 1).drop_duplicates(keep = 'first', subset =['publisher'])
    df_gpc_uspapi_onepage = df_gpc_uspapi.sample(frac=1, random_state=1).drop_duplicates(keep = 'first', subset = ['publisher'])


    df_default_optout = df_default_uspapi_onepage[df_default_uspapi_onepage['us_api_value'].astype(str).str[2] == 'Y']
    df_gpc_optout = df_gpc_uspapi_onepage[df_gpc_uspapi_onepage['us_api_value'].astype(str).str[2] == 'Y']

    #Overlap of those opt-out websites present. 
    df_merged = pd.merge(df_default_optout,df_default_optout, on =['publisher'], how = 'inner')
    n_default_websites = len(df_default_optout) 
    n_gpc_websites = len(df_gpc_optout)

    print(f'In default, % of websites where user opt-out {n_default_websites}/{len(df_default_uspapi_onepage)}({n_default_websites*100/len(df_default_uspapi_onepage)})')
    print(f'In gpc, % of websites where user opt-out {n_gpc_websites}/{len(df_gpc_uspapi_onepage)}({n_gpc_websites*100/len(df_gpc_uspapi_onepage)})')




    #check gpc Enabled field

    df_default_enabled = df_default_uspapi_onepage[df_default_uspapi_onepage.us_api_object.str.contains('gpcEnabled')].copy()
    df_gpc_enabled = df_gpc_uspapi_onepage[df_gpc_uspapi_onepage.us_api_object.str.contains('gpcEnabled')].copy()


    #Get GPC Enabled Field, Parsing Dictionary
    df_default_enabled['gpcEnabled'] = df_default_enabled.us_api_object.str.split('gpcEnabled\":',n=1).str[1].str[:-1]
    #Don't get confuse. I'm using gpc crawl data and gpcEnabled is a separate field inside
    df_gpc_enabled['gpcEnabled'] = df_gpc_enabled.us_api_object.str.split('gpcEnabled\":',n=1).str[1].str[:-1]


    default_enabled_true = df_default_enabled[df_default_enabled['gpcEnabled']=='true']
    gpc_enabled_true = df_gpc_enabled[df_gpc_enabled['gpcEnabled']=='true']
    print(f'Total Number of websites in default where gpcEnabled field was True {len(default_enabled_true)}')
    print(f'Total Number of websites in gpc crawl where gpcEnabled was True {len(gpc_enabled_true)}')

    print(f"Function took {(time.time() - start_time)/60} minutes") 

    """
    We detected the presence of USP API in CA default on 821 websites
    We detected the presence of USP API in CA GPC on 825 websites
    In default, % of websites where user opt-out 24/821(2.92326431181486)
    In gpc, % of websites where user opt-out 380/825(46.06060606060606)
    Total Number of websites in default where gpcEnabled field was True 0
    Total Number of websites in gpc crawl where gpcEnabled was True 48
    Function took 0.0039561708768208826 minutes
    """
def fig_9_http_requests_optout():
    start_time = time.time()
    df_third_parties_privacy_params = pd.read_csv(FIG_9_PLOT_DATA_PATH, sep = '\t')

    FIG_FULL_WIDTH = (12, 3.5)
    # Six colors
    matplotlib.rcParams['axes.prop_cycle'] = cycler('color', ["#5790fc",  "#e42536", "#92dadd",  "#a96b59", "#7a21dd"])

    #Font
    matplotlib.rcParams["pdf.fonttype"] = 42
    matplotlib.rcParams["ps.fonttype"] = 42
    matplotlib.rcParams['axes.axisbelow'] = True

    # Rename the columns in your DataFrame
    DEFAULT_april2023 = "Default (Crawl 1)"
    DEFAULT_jan2024 = "Default (Crawl 4)"
    OPTOUT_GPCNAV_dec2023= "Opt-out GPC (Crawl 3)"
    OPTOUT_TAINTED_API_1YYN_feb24 = "Opt-out USP (Crawl 2)"

    df_plot_data = df_third_parties_privacy_params.rename(columns={
        'percent_default_april2023_optout_thirdparties': DEFAULT_april2023,
        'percent_default_jan2024_optout_thirdparties': DEFAULT_jan2024,
        'percent_gpcnav_dec2023_optout_thirdparties' : OPTOUT_GPCNAV_dec2023,
        'percent_taintedapi_1YYN_feb2024_optout_thirdparties' : OPTOUT_TAINTED_API_1YYN_feb24})


    #Pick top 20 plot data. Already sorted by top 20 third parties in Crawl 1 (default april 2023)
    df_plot_data = df_plot_data.head(20)
    # Initialize the figure and axis
    fig, ax = plt.subplots(2,1,figsize=FIG_FULL_WIDTH, gridspec_kw={'height_ratios':[1.2,4.42]})

    ''''
    First Upper Plot About Count of different third parties present

    '''

    ax[0].legend('',frameon=False)
    ax[0].set_xticks([], []) 
    ax[0].grid(linestyle=':')


    ax[0].spines['right'].set_visible(False)
    ax[0].spines['top'].set_visible(False)
    ax[0].spines['bottom'].set_visible(False)
    ax[0].spines['left'].set_visible(False)

    # Set the width of the bars
    bar_width = 0.17
    grouped = df_plot_data.groupby('domain', sort = False) # Use sort=False to preserve the original order
    # Define the Crawls to Plot
    crawls = ['default_april2023_thirdparties', 'default_jan2024_thirdparties','gpcnav_dec2023_thirdparties', 'taintedapi_1YYN_feb2024_third_parties']
    # Define the x positions which is 20 here since we have 20 third parties
    x = range(len(grouped))
    # Loop through the columns and plot each one,double
    for i, crawl in enumerate(crawls):
        #print(f"crawl {crawl}")
        values = [group[crawl].mean() for name, group in grouped]#for every third party, get their count. we can use mean because there is only one value for every third party
        #for name,group in grouped:
            #print(f"Third Party {name}. Count: {group[crawl].mean()}" )
        ax[0].bar([pos + i * bar_width for pos in x], values, width=bar_width, label=crawl, color = ["#9c9ca1"])
    ax[0].tick_params(which="both", bottom=False, left=False, right=False)
    # Set the x-axis labels
    plt.xticks([pos + bar_width for pos in x], grouped.groups.keys(), rotation=90)
    ax[0].yaxis.grid(True)
    ax[0].xaxis.grid(False)
    ax[0].set_axisbelow(True)
    ax[0].set_yscale('log')
    ax[0].set_ylabel('Count', fontsize=9)
    plt.tight_layout()
    plt.grid(linestyle=':')
    for spine in ('top', 'right', 'bottom', 'left'):
        plt.gca().spines[spine].set_visible(False)

    """
    Plot Bottom
    """
    bar_width = 0.17
    grouped = df_plot_data.groupby('domain', sort = False) # Use sort=False to preserve the original order. domain is already sorted in descending order
    # Define the Crawls to plot
    crawls = [DEFAULT_april2023, DEFAULT_jan2024, OPTOUT_GPCNAV_dec2023,OPTOUT_TAINTED_API_1YYN_feb24]
    # Define the x positions which is 20 here since we have 20 third parties
    x = range(len(grouped))
    # Loop through the crawls e.g DEFAULT_april2023
    for i, crawl in enumerate(crawls):
        #print(f"Plotting {crawl}")
        values = [group[crawl].mean() for name, group in grouped]#for every third party, get opt-out percentage. we can use mean because there is only one value for every third party
        #for name,group in grouped:
            #print(f"Third Party {name} + Count: {group[crawl].mean()}" )
        ax[1].bar([pos + i * bar_width for pos in x], values, width=bar_width, label=crawl)
    # Set the x-axis labels
    plt.xticks([pos + bar_width for pos in x], grouped.groups.keys(), rotation=90)
    ax[1].yaxis.grid(True)
    ax[1].xaxis.grid(False)
    ax[1].set_axisbelow(True)
    ax[1].set_ylabel('Percentage of Opt Out',fontsize = 9)

 
    plt.legend(loc='upper center', bbox_to_anchor=(0.40, 1.05), ncol=2,fontsize='small')

    plt.tight_layout()

    plt.savefig(FIG_PATH + "thirdparties_usprivacy_urls_all.pdf")
    print(f"Function took {(time.time() - start_time)/60} minutes")
    """
    Function took 0.018932851155598958 minutes 
    """
    

def fig_10_tracking_pixels():
    start_time = time.time()
    #LOAD DATA
    df_all_tracking_pixels = pd.read_csv(FIG_10_PLOT_DATA_PATH, sep = '\t')
    df_default_april2023_uspapi = pd.read_csv(CA_DEFAULT_USP_API_PATH, sep = '\t')
    df_default_jan2024_uspapi = pd.read_csv(CA_DEFAULT_2024_USP_API_PATH, sep = '\t')

    df_default_april2023_uspapi = df_default_april2023_uspapi[df_default_april2023_uspapi['us_api_value'].notnull()]
    #Get unique publishers present in Default April 2023 Crawl.
    df_default_april2023_uspapi_publishers = df_default_april2023_uspapi.drop_duplicates(subset = ['publisher'])
    df_default_april2023_uspapi_publishers = df_default_april2023_uspapi_publishers[['publisher']]
    #Get unique publishers present in Jan 2024 Crawl.
    df_default_jan2024_uspapi = df_default_jan2024_uspapi[df_default_jan2024_uspapi['us_api_value'].notnull()]
    #Get unique publishers present in Default April 2023 Crawl.
    df_default_jan2024_uspapi_publishers = df_default_jan2024_uspapi.drop_duplicates(subset = ['publisher'])
    df_default_jan2024_uspapi_publishers = df_default_jan2024_uspapi_publishers[['publisher']]

    DEFAULT_april2023 = "Crawl 1"
    DEFAULT_jan2024 = "Crawl 4"
    OPTOUT_GPCNAV_dec2023= "Crawl 3"
    OPTOUT_TAINTED_API_1YYN_feb24 = "Crawl 2"

    req_cols = ['website','default_april2023_trackingpixels','default_jan2024_trackingpixels','gpc_dec2023_trackingpixels','taintedapi_1YYN_feb2024_trackingpixels']
    df_all_tracking_pixels_cleaned = df_all_tracking_pixels[req_cols]
    #Get only those websites that were successfully crawled in all four experiments.
    df_all_tracking_pixels_cleaned = df_all_tracking_pixels_cleaned[(df_all_tracking_pixels_cleaned!=-1).all(axis=1)]

    #Get only those websites which are present in the Default April 2023 Crawl and Default Jan 2024.
    df_unique_publishers = pd.merge(df_default_april2023_uspapi_publishers, df_default_jan2024_uspapi_publishers, how = 'outer', on ='publisher')

     #Generate DataFrame for pixels 1) that were crawled successfully in all experiments. 2) had adopted USP API in Default April 2023 and Default Jan 2024
    df_plot_tracking_pixels = pd.merge(df_all_tracking_pixels_cleaned,df_unique_publishers, how = 'inner', left_on=['website'],right_on = ['publisher'] )
    print(f'Total Number of websites 1) that were crawled successfully in all experiments. 2) had adopted USP API in Default April 2023 and Default Jan 2024 {df_plot_tracking_pixels.publisher.nunique()}')
    df_plot_tracking_pixels = df_plot_tracking_pixels.rename(columns={
    'default_april2023_trackingpixels': DEFAULT_april2023,
    'default_jan2024_trackingpixels': DEFAULT_jan2024,
    'gpc_dec2023_trackingpixels': OPTOUT_GPCNAV_dec2023,
    'taintedapi_1YYN_feb2024_trackingpixels': OPTOUT_TAINTED_API_1YYN_feb24
})
    plt.rcParams["pdf.fonttype"] = 42
    plt.rcParams["ps.fonttype"] = 42
    plt.rcParams['axes.axisbelow'] = True
    plt.rcParams['axes.prop_cycle'] = cycler('color', ["#5790fc", "#f89c20", "#e42536", "#964a8b", "#9c9ca1", "#7a21dd"])

    FIG_HALF_WIDTH = (5, 3)
    fig = plt.figure(figsize=FIG_HALF_WIDTH)

    ax = fig.add_subplot(111)
    ax.tick_params(which="both", bottom=False, left=False, right=False)
    plt.grid(linestyle=':')
    for spine in ('top', 'right', 'bottom', 'left'):
        plt.gca().spines[spine].set_visible(False)

    plt.rcParams['axes.axisbelow'] = True
    experiments = [DEFAULT_april2023,OPTOUT_TAINTED_API_1YYN_feb24, OPTOUT_GPCNAV_dec2023,DEFAULT_jan2024]
    ax.set_xticklabels(experiments, fontsize=8)

    plt.ylabel("Number of Tracking Requests ")
    boxplot = ax.boxplot([df_plot_tracking_pixels[DEFAULT_april2023],df_plot_tracking_pixels[OPTOUT_TAINTED_API_1YYN_feb24], df_plot_tracking_pixels[OPTOUT_GPCNAV_dec2023], df_plot_tracking_pixels[DEFAULT_jan2024]])
    plt.tight_layout() 
    plt.savefig(FIG_PATH + "boxplot_tracking_comparison_websites_api_presence.pdf")

    #remove 'publisher and website columns' not needed, and keep only column with total number of pixels
    columns_to_remove = ['website', 'publisher']
    df_pixel_experiment_stats = df_plot_tracking_pixels.drop(columns = columns_to_remove)
    #Create a list of experiments excluding DEFAULT_april2023  
    experiments = [col for col in df_pixel_experiment_stats.columns if col != DEFAULT_april2023]

    """
     1) Get p-values and apply Two tests across different crawls  (https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.ttest_rel.html)
     2) Calculate efect size using Cohen's 
    """
    # Create an empty DataFrame to store the results
    results_df = pd.DataFrame(columns=['Experiment', 'Baseline', 'T-Statistic', 'P-Value', 'Effect Size'])

    # Perform t-test and calculate effect size for each experiment relative to DEFAULT_april2023
    for experiment in experiments:
        t_statistic, p_value = stats.ttest_rel(df_pixel_experiment_stats[DEFAULT_april2023], df_pixel_experiment_stats[experiment])
        effect_size = str(cohen_d(df_pixel_experiment_stats[DEFAULT_april2023], df_pixel_experiment_stats[experiment]))    

        new_row = pd.DataFrame([{'Experiment': experiment,
                         'Baseline': DEFAULT_april2023, 
                         'T-Statistic': t_statistic,
                         'P-Value': p_value,
                         'Effect Size': effect_size}])
        results_df = pd.concat([results_df, new_row], ignore_index=True)

 
    
    df_statistical_significant = results_df[results_df['P-Value']<=0.05]
    print(results_df)
    print('Statisitcal Significant  Results below <=0.05')
    print(df_statistical_significant)
    #Save Results
    results_df.to_latex(FIG_PATH +"tracking_pixels_t_test_and_effect_size_results.tex", index = False)
    print(f"Function took {(time.time() - start_time)/60} minutes") 

    """
   Experiment Baseline  T-Statistic   P-Value           Effect Size
    0    Crawl 4  Crawl 1    -0.520639  0.602764  -0.02057906751138179
    1    Crawl 3  Crawl 1    -2.920558  0.003593  -0.09697114405020006
    2    Crawl 2  Crawl 1     1.817478  0.069522   0.06843231054255337
    Statisitcal Significant  Results below <=0.05
    Experiment Baseline  T-Statistic   P-Value           Effect Size
    1    Crawl 3  Crawl 1    -2.920558  0.003593  -0.09697114405020006
    """




parser = argparse.ArgumentParser(description = "Generating Figs and Tables and Analysis")
section_1 = ["tab_3_param_consent_table", "fig_1_uspapi_ranks", "fig_2_uspapi_category","tab_2_cookie_count","generate_uspapi_adoption_rate_different_ccpa_criteria_numbers","generate_usprivacy_cookies_numbers","generate_uspapi_cookie_together_numbers"]
section_2 = ["fig_3_uspapi_writing", "fig_4_uspapi_reading","fig_5_uspapi_consistency_read", "fig_6_usprivacycookie_writing"]
section_3 = ["generate_taint_crawl_validation_numbers", "generate_default_chain_numbers", "tab_4_5_6_generate_initiator_receiver_tables", "fig_7_initiator_receiver_consistency", "fig_8_top_receivers_params"]
section_4 = [ "generate_default_gpc_optout_numbers", "fig_9_http_requests_optout","fig_10_tracking_pixels","generate_fig_tables","generate_analysis"]
parser.add_argument('-f','--function',type = str, choices = section_1 + section_2+ section_3 + section_4, required = True, help = 'Generate Plots and Tables')


args =parser.parse_args()
##########################################################################
#Section 4.1
if args.function == "fig_1_uspapi_ranks":
    print(f"Generating Figure 1")
    fig_1_uspapi_ranks()
elif args.function == "tab_2_cookie_count":
    print(f"Generating Table 2")
    tab_2_cookie_count()
elif args.function =="fig_2_uspapi_category":
    print(f"Generating Figure 2")
    fig_2_uspapi_category()
elif args.function== "tab_3_param_consent_table":
    print(f"Generating Table 3")
    tab_3_param_consent_table()
elif args.function =="generate_uspapi_adoption_rate_different_ccpa_criteria_numbers":
    generate_uspapi_adoption_rate_different_ccpa_criteria_numbers()
elif args.function =="generate_usprivacy_cookies_numbers":
    generate_usprivacy_cookies_numbers()
elif args.function == "generate_uspapi_cookie_together_numbers":
    generate_uspapi_cookie_together_numbers()
##########################################################################
#Section 4.2
elif args.function == "fig_3_uspapi_writing":
    print(f"Generating Figure 3")
    fig_3_uspapi_writing()
elif args.function == "fig_4_uspapi_reading":
    print(f"Generating Figure 4")
    fig_4_uspapi_reading()
elif args.function =="fig_5_uspapi_consistency_read":
    print(f"Generating Figure 5")
    fig_5_uspapi_consistency_read()
elif args.function =="fig_6_usprivacycookie_writing":
    print(f"Generating Figure 6")
    fig_6_usprivacycookie_writing()
##########################################################################
#SECTION 4.3
elif args.function == "generate_taint_crawl_validation_numbers":
    generate_taint_crawl_validation_numbers()
elif args.function == "tab_4_5_6_generate_initiator_receiver_tables":
    print(f"Generating Table 4, 5, 6")
    tab_4_5_6_generate_initiator_receiver_tables()
elif args.function =="fig_7_initiator_receiver_consistency":
    print(f"Generating Figure 7")
    fig_7_initiator_receiver_consistency()
elif args.function== "fig_8_top_receivers_params":
    print(f"Generating Figure 8")
    fig_8_top_receivers_params()
elif args.function =="generate_default_chain_numbers":
   generate_default_chain_numbers() 
###########################################################################
#SECTION 4.4
elif args.function == "generate_default_gpc_optout_numbers":
    generate_default_gpc_optout_numbers()
elif args.function == "fig_9_http_requests_optout":
    print(f"Generating Figure 9")
    fig_9_http_requests_optout()
elif args.function =="fig_10_tracking_pixels":
    print(f"Generating Figure 10") 
    fig_10_tracking_pixels()
elif args.function == "generate_fig_tables":



    print(f"Generate Figures")
    start_time = time.time()
    # Generate Figures
    print("Generate Fig 1")
    fig_1_uspapi_ranks()
    print("Generate Fig 2")
    fig_2_uspapi_category()
    print("Generate Fig 3")
    fig_3_uspapi_writing()
    print("Generate Fig 4")
    fig_4_uspapi_reading()
    print("Generate Fig 5")
    fig_5_uspapi_consistency_read()
    print("Generate Fig 6")
    fig_6_usprivacycookie_writing()
    print("Generate Fig 7")
    fig_7_initiator_receiver_consistency()
    print("Generate Fig 8")
    fig_8_top_receivers_params()
    print("Generate Fig 9")
    fig_9_http_requests_optout()
    print("Generate Fig 10")
    fig_10_tracking_pixels()

    #Generating Tables
    print("Generate Table 2")
    tab_2_cookie_count()
    print("Generate Table 3")
    tab_3_param_consent_table()
    print("Generate Tabel 4, 5, 6")
    tab_4_5_6_generate_initiator_receiver_tables()
    print(f"It took {(time.time() - start_time)/60} minutes")
elif args.function == "generate_analysis":
   print(f"Generate Different Adoption Rate")
   generate_uspapi_adoption_rate_different_ccpa_criteria_numbers() 
   print(f"Generate Cookie Numbers")
   generate_usprivacy_cookies_numbers()
   print(f"generate_uspapi_cookie_together_numbers")
   generate_uspapi_cookie_together_numbers()
   print(f"generate_taint_crawl_validation_numbers")
   generate_taint_crawl_validation_numbers()
   print(f"generate_default_chain_numbers")
   generate_default_chain_numbers()
   print(f"generate_default_gpc_optout_numbers")
   generate_default_gpc_optout_numbers()
