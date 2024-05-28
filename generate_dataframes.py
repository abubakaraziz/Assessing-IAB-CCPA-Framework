import argparse
import pandas as pd
import time
from file_paths import *



def generate_us_privacy_params():
    start_time = time.time()
    df_default_april2023_adchains = pd.read_csv(CA_DEFAULT_HTTP_REQUESTS_CHAINS_PATH, sep = '\t')
    df_default_jan2024_adchains   = pd.read_csv(CA_DEFAULT_2024_HTTP_REQUESTS_CHAINS_PATH, sep = '\t')
    df_gpcnav_dec2023_adchains = pd.read_csv(CA_GPC_HTTP_REQUESTS_CHAINS_PATH, sep = '\t') 
    df_taintedapi_1YYN_feb2024_adchains = pd.read_csv(CA_1YYN_HTTP_REQUESTS_CHAINS_PATH, sep = '\t')

    #Third Parties Present with US Privacy parameters 
    df_default_april2023_usprivacy_params = df_default_april2023_adchains[df_default_april2023_adchains['is_consent_signal_present']==True]
    df_default_jan2024_usprivacy_params = df_default_jan2024_adchains[df_default_jan2024_adchains['is_consent_signal_present']==True]
    df_gpcnav_dec2023_usprivacy_params = df_gpcnav_dec2023_adchains[df_gpcnav_dec2023_adchains['is_consent_signal_present']==True]
    df_taintedapi_1YYN_feb2024_usprivacy_params = df_taintedapi_1YYN_feb2024_adchains[df_taintedapi_1YYN_feb2024_adchains['is_consent_signal_present']==True] 


    #Filter Out URLs where we detect Opt-Out usprivacy parameters
    df_default_april2023_optout_thirdparties = df_default_april2023_usprivacy_params[df_default_april2023_usprivacy_params['consent_signal_value'].str[2] == 'Y']
    df_default_jan2024_optout_thirdparties = df_default_jan2024_usprivacy_params[df_default_jan2024_usprivacy_params['consent_signal_value'].str[2] == 'Y'] 
    df_gpcnav_dec2023_optout_thirdparties = df_gpcnav_dec2023_usprivacy_params[df_gpcnav_dec2023_usprivacy_params['consent_signal_value'].str[2] == 'Y']      
    df_taintedapi_1YYN_feb2024_optout_thirdparties =  df_taintedapi_1YYN_feb2024_usprivacy_params[df_taintedapi_1YYN_feb2024_usprivacy_params['consent_signal_value'].str[2]=='Y']

    #Count OptOut ThirdParties
    df_default_april2023_optout_thirdparties = df_default_april2023_optout_thirdparties.groupby(['domain'])['url'].count().reset_index().rename(columns={'url':'default_april2023_optout_thirdparties'})
    df_default_jan2024_optout_thirdparties  = df_default_jan2024_optout_thirdparties.groupby(['domain'])['url'].count().reset_index().rename(columns={'url':'default_jan2024_optout_thirdparties'}) 
    df_gpcnav_dec2023_optout_thirdparties = df_gpcnav_dec2023_optout_thirdparties.groupby(['domain'])['url'].count().reset_index().rename(columns={'url':'gpc_dec2023_optout_thirdparties'})  
    df_taintedapi_1YYN_feb2024_optout_thirdparties = df_taintedapi_1YYN_feb2024_optout_thirdparties.groupby(['domain'])['url'].count().reset_index().rename(columns={'url':'taintedapi_1YYN_feb2024_optout_thirdparties'})


    #All Third Parties Count. 
    df_default_april2023_third_parties = df_default_april2023_adchains.groupby(['domain'])['url'].count().reset_index().rename(columns={'url':'default_april2023_thirdparties'})
    df_default_jan2024_third_parties = df_default_jan2024_adchains.groupby(['domain'])['url'].count().reset_index().rename(columns={'url':'default_jan2024_thirdparties'}) 
    df_gpcnav_dec2023_third_parties = df_gpcnav_dec2023_adchains.groupby(['domain'])['url'].count().reset_index().rename(columns={'url':'gpcnav_dec2023_thirdparties'}) 
    df_taintedapi_1YYN_feb2024_third_parties =  df_taintedapi_1YYN_feb2024_adchains.groupby(['domain'])['url'].count().reset_index().rename(columns={'url': 'taintedapi_1YYN_feb2024_third_parties'})



    # Combine OptOut ThirdParties dataframes
    df_optout_thirdparties = pd.merge(df_default_april2023_optout_thirdparties, df_default_jan2024_optout_thirdparties, how='outer', on='domain')
    df_optout_thirdparties = pd.merge(df_optout_thirdparties, df_gpcnav_dec2023_optout_thirdparties, how='outer', on='domain')
    df_optout_thirdparties = pd.merge(df_optout_thirdparties, df_taintedapi_1YYN_feb2024_optout_thirdparties, how='outer', on='domain')



    # Combine All Third Parties Count dataframes
    df_third_parties = pd.merge(df_default_april2023_third_parties, df_default_jan2024_third_parties, how='outer', on='domain')
    df_third_parties = pd.merge(df_third_parties, df_gpcnav_dec2023_third_parties, how='outer', on='domain')
    df_third_parties = pd.merge(df_third_parties, df_taintedapi_1YYN_feb2024_third_parties, how='outer', on='domain')
 

    # Combine OptOut ThirdParties and All Third Parties Count dataframes
    df_combined = pd.merge(df_optout_thirdparties, df_third_parties, how='outer', on='domain')
    # Fill NaN values with 0
    df_combined = df_combined.fillna(0)

    # Calculate percentage of opt-outs for each category
    df_combined['percent_default_april2023_optout_thirdparties'] = df_combined['default_april2023_optout_thirdparties'] * 100 / df_combined['default_april2023_thirdparties']
    df_combined['percent_default_jan2024_optout_thirdparties'] = df_combined['default_jan2024_optout_thirdparties'] * 100 / df_combined['default_jan2024_thirdparties']
    df_combined['percent_gpcnav_dec2023_optout_thirdparties'] = df_combined['gpc_dec2023_optout_thirdparties'] * 100 / df_combined['gpcnav_dec2023_thirdparties']
    df_combined['percent_taintedapi_1YYN_feb2024_optout_thirdparties'] = df_combined['taintedapi_1YYN_feb2024_optout_thirdparties'] * 100 / df_combined['taintedapi_1YYN_feb2024_third_parties']
   



    # Reorder columns
    df_combined = df_combined[['domain','default_april2023_thirdparties', 'default_april2023_optout_thirdparties','percent_default_april2023_optout_thirdparties', 
                                'default_jan2024_thirdparties','default_jan2024_optout_thirdparties', 'percent_default_jan2024_optout_thirdparties',
                                'gpcnav_dec2023_thirdparties','gpc_dec2023_optout_thirdparties', 'percent_gpcnav_dec2023_optout_thirdparties',
                                'taintedapi_1YYN_feb2024_third_parties','taintedapi_1YYN_feb2024_optout_thirdparties', 'percent_taintedapi_1YYN_feb2024_optout_thirdparties'
                               ]]
    # Round the values to two decimal places
    df_combined = df_combined.round(2)

    
    #Filter results by ad-networks that have most number of URLs in Default Crawl and pick top 20 and Drop all ad-networks which have <=1% optout in tainted api crawl.
    df_combined_filtered_usapi= df_combined[df_combined['percent_taintedapi_1YYN_feb2024_optout_thirdparties']>=1].sort_values(ascending = False, by = ['default_april2023_thirdparties'])
    
    #Filter Out Top third parties that have <= 1 US privacy parameter
    third_parties_no_privacy_urls = df_combined[df_combined['percent_taintedapi_1YYN_feb2024_optout_thirdparties']<1].sort_values(ascending = False, by = ['taintedapi_1YYN_feb2024_third_parties']).rename(columns= {"domain":"Domain","taintedapi_1YYN_feb2024_third_parties":"Count","percent_taintedapi_1YYN_feb2024_optout_thirdparties":"Percent"})
    third_parties_no_privacy_urls.to_csv(DATA_PATH + "less_than_1_percent_privacy_urls.tsv", sep = '\t', index = False) 
    df_combined_filtered_usapi.to_csv(DATA_PATH + "fig_9_thirdparties_usprivacy_urls_all.tsv", sep = '\t', index = False) 
    print(f"Function took {(time.time() - start_time)/60} minutes") 
    """
    Function took 4.5276761770248415 minutes
    """



'''
Generate Tracking Pixels only and save them.
'''

def get_tracking_pixels_websites(experiment_name, df_ad_chains, tracking_col_name):

    #Get Unique Chains Stats
    print(f'Total Number of unique chains in {experiment_name} {df_ad_chains.chainid.nunique()}')
    print(f'Total Number of unique websites with adchains that has USP API in {experiment_name} {df_ad_chains.website.nunique()}')


    ############## FOR FILTERING TRACKING PIXEL tracking, Filter out the TRACKING pixel and then pick the last node because last node gets loaded successfully. 
    df_trackingpixels = df_ad_chains[df_ad_chains['is_tracking_pixel']==True]
    df_trackingpixels = df_trackingpixels.sort_values('pos', ascending=False).drop_duplicates(['chainid'])

    print(f'Total Number of tracking pixels in {experiment_name} {len(df_trackingpixels)}')
    print(f'Number of unique websites in {experiment_name} with tracking pixels {df_trackingpixels.website.nunique()}')


    
    #Count Tracking Pixels Per Website
    df_websitetracking = df_trackingpixels.groupby(['website'])['url'].count().reset_index().rename(columns = {'url':tracking_col_name})
    df_websitetracking = pd.merge(df_ad_chains[['website']].drop_duplicates(subset=['website']),df_websitetracking, how = 'left', on ='website')
    df_websitetracking[tracking_col_name] = df_websitetracking[tracking_col_name].fillna(0)
    return df_websitetracking



def generate_tracking_pixels_dataframe():
    start_time = time.time()
    df_default_april2023_adchains = pd.read_csv(CA_DEFAULT_HTTP_REQUESTS_CHAINS_PATH, sep = '\t')
    df_default_jan2024_adchains   = pd.read_csv(CA_DEFAULT_2024_HTTP_REQUESTS_CHAINS_PATH, sep = '\t')
    df_gpcnav_dec2023_adchains = pd.read_csv(CA_GPC_HTTP_REQUESTS_CHAINS_PATH, sep = '\t') 
    df_taintedapi_1YYN_feb2024_adchains = pd.read_csv(CA_1YYN_HTTP_REQUESTS_CHAINS_PATH, sep = '\t')

    # Rename the columns in your DataFrame
    DEFAULT_april2023 = "Default April, 23"
    DEFAULT_jan2024 = "Default Jan, 24"
    OPTOUT_GPCNAV_dec2023= "Opt-out GPC Nav Dec, 23"
    OPTOUT_TAINTED_API_1YYN_feb24 =  "Opt-out Tainted API Feb, 24"

    #Merge Count of Different Tracking Pixels Together
    df_default_april2023_websitetracking = get_tracking_pixels_websites(experiment_name=DEFAULT_april2023,df_ad_chains=df_default_april2023_adchains,tracking_col_name='default_april2023_trackingpixels')
    df_default_jan2024_websitetracking = get_tracking_pixels_websites(experiment_name=DEFAULT_jan2024,df_ad_chains=df_default_jan2024_adchains,tracking_col_name='default_jan2024_trackingpixels')
    df_gpcnav_dec2023_websitetracking = get_tracking_pixels_websites(experiment_name=OPTOUT_GPCNAV_dec2023,df_ad_chains=df_gpcnav_dec2023_adchains,tracking_col_name='gpc_dec2023_trackingpixels')  
    df_taintedapi_1YYN_feb2024_websitetracking =  get_tracking_pixels_websites(experiment_name=OPTOUT_TAINTED_API_1YYN_feb24,df_ad_chains=df_taintedapi_1YYN_feb2024_adchains,tracking_col_name='taintedapi_1YYN_feb2024_trackingpixels') 

    # Merge data frames
    merged_df = pd.merge(df_default_april2023_websitetracking, df_default_jan2024_websitetracking, 
                     on='website', how='outer') 
    merged_df = pd.merge(merged_df, df_gpcnav_dec2023_websitetracking, 
                     on='website', how='outer')
    merged_df = pd.merge(merged_df, df_taintedapi_1YYN_feb2024_websitetracking, 
                     on='website', how='outer')
    # Fill null values with -1 because null values mean that particular website wasn't crawled successfully and hence we have no idea how many tracking pixels were present.
    merged_df = merged_df.fillna(-1)

    merged_df.to_csv(DATA_PATH + "fig_10_tracking_pixels_all_websites.tsv", sep = '\t', index = False)
    print(f"Function took {(time.time() - start_time)/60} minutes") 

    """
    Total Number of unique chains in Default April, 23 3102021
    Total Number of unique websites with adchains that has USP API in Default April, 23 6799
    Total Number of tracking pixels in Default April, 23 530253
    Number of unique websites in Default April, 23 with tracking pixels 5848
    Total Number of unique chains in Default Jan, 24 2903480
    Total Number of unique websites with adchains that has USP API in Default Jan, 24 6478
    Total Number of tracking pixels in Default Jan, 24 517867
    Number of unique websites in Default Jan, 24 with tracking pixels 5316
    Total Number of unique chains in Opt-out GPC Nav Dec, 23 3196301
    Total Number of unique websites with adchains that has USP API in Opt-out GPC Nav Dec, 23 6356
    Total Number of tracking pixels in Opt-out GPC Nav Dec, 23 551389
    Number of unique websites in Opt-out GPC Nav Dec, 23 with tracking pixels 5308
    Total Number of unique chains in Opt-out Tainted API Feb, 24 3437455
    Total Number of unique websites with adchains that has USP API in Opt-out Tainted API Feb, 24 6900
    Total Number of tracking pixels in Opt-out Tainted API Feb, 24 551178
    Number of unique websites in Opt-out Tainted API Feb, 24 with tracking pixels 5732
    Function took 4.841628058751424 minutes
    """

    
def generate_stats_adchain_dataframe(experiment, fp_ad_dataframe):
    df_ad_chains_parsed = pd.read_csv(fp_ad_dataframe, sep = '\t')
    nwebsites_atleast_onechain = df_ad_chains_parsed.website.nunique()
    return pd.DataFrame([{'experiment':experiment, 'nwebsites_atleast_onechain':nwebsites_atleast_onechain}])




parser = argparse.ArgumentParser(description = "Generating DataFrame and Stats")
parser.add_argument('-f','--function',type = str, choices = [ "generate_stats_adchain_dataframe","generate_us_privacy_params","generate_tracking_pixels_dataframe"], required = True, help = 'Functions for different Vertical Plots')
args =parser.parse_args()

 
if args.function == "generate_stats_adchain_dataframe":
    rows_list = []
    #CA Experiment
    df_ad_frame = pd.DataFrame()
    EXPERIMENTS =["crawl1","crawl2", "crawl3","crawl4"]
    fp_nwebsites_ad_chains = FIG_PATH + "tab_1_nwebsites_ad_chains.tsv" 
    for EXPERIMENT in EXPERIMENTS:
        fp_working_dir = DATA_PATH + EXPERIMENT + "/http_initiator_requests.tsv"
        df_ad_chain_stats = generate_stats_adchain_dataframe(EXPERIMENT,fp_working_dir)
        df_ad_frame = pd.concat([df_ad_frame, df_ad_chain_stats], ignore_index=True) 
    #Save Dataframe              
    df_ad_frame.to_csv(fp_nwebsites_ad_chains, sep = '\t', index = False)
elif args.function == "generate_us_privacy_params":
    generate_us_privacy_params()

elif args.function =="generate_tracking_pixels_dataframe":
    generate_tracking_pixels_dataframe()


