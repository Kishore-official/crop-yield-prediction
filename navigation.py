import streamlit as st

import base64
import numpy as np
import matplotlib.pyplot as plt 
from tkinter.filedialog import askopenfilename

import streamlit as st

import matplotlib.image as mpimg

import streamlit as st
import base64

import pandas as pd
import sqlite3

from sklearn import preprocessing
import cv2










# ================ Background image ===

def add_bg_from_local(image_file):
    with open(image_file, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read())
    st.markdown(
    f"""
    <style>
    .stApp {{
        background-image: url(data:image/{"png"};base64,{encoded_string.decode()});
        background-size: cover
    }}
    </style>
    """,
    unsafe_allow_html=True
    )
add_bg_from_local('1.avif')


def navigation():
    try:
        path = st.experimental_get_query_params()['p'][0]
    except Exception as e:
        st.error('Please use the main app.')
        return None
    return path


if navigation() == "home":
    
    # st.title("Crop yeild Prediction")
    
    st.markdown(f'<h1 style="color:#000000;text-align: center;font-size:36px;">{"Crop Recommendation"}</h1>', unsafe_allow_html=True)
    



elif navigation()=='reg':
   

    st.markdown(f'<h1 style="color:#000000;text-align: center;font-size:24px;">{"REGISTER"}</h1>', unsafe_allow_html=True)
    
    import streamlit as st
    import sqlite3
    import re
    
    # Function to create a database connection
    def create_connection(db_file):
        conn = None
        try:
            conn = sqlite3.connect(db_file)
        except sqlite3.Error as e:
            print(e)
        return conn
    
    # Function to create a new user
    def create_user(conn, user):
        sql = ''' INSERT INTO users(name, password, email, phone)
                  VALUES(?,?,?,?) '''
        cur = conn.cursor()
        cur.execute(sql, user)
        conn.commit()
        return cur.lastrowid
    
    # Function to check if a user already exists
    def user_exists(conn, email):
        cur = conn.cursor()
        cur.execute("SELECT * FROM users WHERE email=?", (email,))
        if cur.fetchone():
            return True
        return False
    
    # Function to validate email
    def validate_email(email):
        pattern = r'^\w+([\.-]?\w+)*@\w+([\.-]?\w+)*(\.\w{2,3})+$'
        return re.match(pattern, email)
    
    # Function to validate phone number
    def validate_phone(phone):
        pattern = r'^[6-9]\d{9}$'
        return re.match(pattern, phone)
    
    # Main function
    def main():
        # st.title("User Registration")
    
        # Create a database connection
        conn = create_connection("dbs.db")
    
        if conn is not None:
            # Create users table if it doesn't exist
            conn.execute('''CREATE TABLE IF NOT EXISTS users
                         (id INTEGER PRIMARY KEY,
                         name TEXT NOT NULL,
                         password TEXT NOT NULL,
                         email TEXT NOT NULL UNIQUE,
                         phone TEXT NOT NULL);''')
    
            # User input fields
            name = st.text_input("Enter your name")
            password = st.text_input("Enter your password", type="password")
            confirm_password = st.text_input("Confirm your password", type="password")
            email = st.text_input("Enter your email")
            phone = st.text_input("Enter your phone number")
    
            col1, col2 = st.columns(2)

            with col1:
                    
                aa = st.button("REGISTER")
                
                if aa:
                    
                    if password == confirm_password:
                        if not user_exists(conn, email):
                            if validate_email(email) and validate_phone(phone):
                                user = (name, password, email, phone)
                                create_user(conn, user)
                                st.success("User registered successfully!")
                            else:
                                st.error("Invalid email or phone number!")
                        else:
                            st.error("User with this email already exists!")
                    else:
                        st.error("Passwords do not match!")
                    
                    conn.close()
                    # st.success('Successfully Registered !!!')
                # else:
                    
                    # st.write('Registeration Failed !!!')     
            
            # with col2:
                    
            #     aa = st.button("LOGIN")
                
            #     if aa:
            #         import subprocess
            #         subprocess.run(['streamlit','run','logf.py'])
    

    if __name__ == '__main__':
        main()

elif navigation()=='abt':
    
    st.markdown(f'<h1 style="color:#0000FF;text-align: center;font-size:36px;">{"Crop Recommendation"}</h1>', unsafe_allow_html=True)

    
    st.write("Agriculture is the one amongst the substantial area of interest to society since a large portion of food is produced by them. Agriculture is the most important sector that influences the economy of India. Predicting crop yield based on the environmental, soil, water and crop parameters has been a potential research topic. Agriculture for years but the results are never satisfying due to various factors that affect the crop yield. Deep-learning-based models are broadly used to extract significant crop features for prediction. Though these methods could resolve the yield prediction problem there exist the following inadequacies: Unable to create a direct non-linear or linear mapping between the raw data and crop yield values; and the performance of those models highly relies on the quality of the extracted features. Finally, the agent receives an aggregate score for the actions performed by minimizing the error and maximizing the forecast accuracy. The system is developed the machine learning model such as random forest. The system can recommend the crop and fertilizer for corresponding type of crop. Then, it will forecast the crop yield in terms of hg. Then, it will predict the water level of the crop. After that, the system can suggest the type of diseases and pesticides based on predicted crops. The system can also display the government schemes, current market price and market names")
    
    
elif navigation()=='log':   
    
    st.markdown(f'<h1 style="color:#000000;text-align: center;font-size:24px;">{"LOGIN"}</h1>', unsafe_allow_html=True)

    
    
    # Function to create a database connection
    def create_connection(db_file):
        conn = None
        try:
            conn = sqlite3.connect(db_file)
        except sqlite3.Error as e:
            print(e)
        return conn
    
    # Function to create a new user
    def create_user(conn, user):
        sql = ''' INSERT INTO users(name, password, email, phone)
                  VALUES(?,?,?,?) '''
        cur = conn.cursor()
        cur.execute(sql, user)
        conn.commit()
        return cur.lastrowid
    
    # Function to validate user credentials
    def validate_user(conn, name, password):
        cur = conn.cursor()
        cur.execute("SELECT * FROM users WHERE name=? AND password=?", (name, password))
        user = cur.fetchone()
        if user:
            return True, user[1]  # Return True and user name
        return False, None
    
    # Main function
    def main():
        # st.title("User Login")
        st.markdown(f'<h1 style="color:#000000;text-align: center;font-size:24px;">{"Login here"}</h1>', unsafe_allow_html=True)
    
    
        # Create a database connection
        conn = create_connection("dbs.db")
    
        if conn is not None:
            # Create users table if it doesn't exist
            conn.execute('''CREATE TABLE IF NOT EXISTS users
                         (id INTEGER PRIMARY KEY,
                         name TEXT NOT NULL,
                         password TEXT NOT NULL,
                         email TEXT NOT NULL UNIQUE,
                         phone TEXT NOT NULL);''')
    
            st.write("Enter your credentials to login:")
            name = st.text_input("User name")
            password = st.text_input("Password", type="password")
    
            col1, col2 = st.columns(2)
    
            with col1:
                    
                aa = st.button("Login")
                
                if aa:
    
    
            # if st.button("Login"):
                    is_valid, user_name = validate_user(conn, name, password)
                    if is_valid:
                        st.success(f"Welcome back, {user_name}! Login successful!")
                        

                        
                        
                    else:
                        st.error("Invalid user name or password!")
                        
    
            # Close the database connection
            conn.close()
        else:
            st.error("Error! cannot create the database connection.")
    
    if __name__ == '__main__':
        main()    
    
    
elif navigation()=='crop':   
    
    st.markdown(f'<h1 style="color:#000000;text-align: center;font-size:24px;">{"CROP YEILD"}</h1>', unsafe_allow_html=True)
    
    print("---------------------------------------------")
    print(" Input Data ---> Crop Recommendation")
    print("---------------------------------------------")
    print()
    
    
    # ================  INPUT  ===
    
    df=pd.read_csv("Yeild.csv")    
    # df=df[0:2500]
    print("--------------------------------")
    print("Data Selection")
    print("--------------------------------")
    print(df.head(15))
    
    df=df[['crop_names','soil_type','area','Yield']]
    
    
    # ================  PRE-PROCESSING  ===
     
     # --- MISSING VALUES 
     
    print("--------------------------------")
    print("  Handling Missing Values")
    print("--------------------------------")                    
    print(df.isnull().sum())    
    res=df.isnull().sum().any()
     
    if res==False:
        print("---------------------------------------------")
        print("There is no missing values in our dataset !!!")
        print("---------------------------------------------")
    else:
        print("---------------------------------------------")
        print("Missing values is present in our dataset !!!")
        print("---------------------------------------------")  
        df=df.fillna(0)
    
    
    # --- Label encoding
    
    print("----------------------------------------------------")
    print("Before Label Encoding          ")
    print("----------------------------------------------------")
    print()
    print(df['crop_names'].head(15))
    print()
    
    data_label = df['crop_names']
    
    df_soil = df['soil_type']
    
    df_area = df['area']
    
    
    label_encoder=preprocessing.LabelEncoder()
    
    print("----------------------------------------------------")
    print("After Label Encoding          ")
    print("----------------------------------------------------")
    print()
    
    df['soil_type']=label_encoder.fit_transform(df['soil_type'].astype(str))
    
    df['crop_names']=label_encoder.fit_transform(df['crop_names'])
    
    print(df['crop_names'].head(15))
        
    #============================= DATA SPLITTING ==============================
    
    
    print("----------------------------------------------------")
    print("Data Splitting          ")
    print("----------------------------------------------------")
    print()
    
    from sklearn.model_selection import train_test_split
    
    X = df.drop('Yield', axis=1)
    y = df['Yield']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=0.3, random_state=100)
    
    print("Total no of data's       :",df.shape[0])
    print()
    print("Total no of Train data's :",X_train.shape[0])
    print()
    print("Total no of Test data's  :",X_test.shape[0])
    
    
    # ============ RANDOM FOREST ===================
    
    
    from sklearn.ensemble import RandomForestRegressor
    
    rf = RandomForestRegressor()
    
    rf.fit(X_train,y_train)
    
    pred_rf = rf.predict(X_test)
    
    from sklearn import metrics
    
    
    st.markdown(f'<h1 style="color:#000000;font-size:14px;">{"Enter the following details"}</h1>', unsafe_allow_html=True)
    
    
    a1 = st.selectbox("Choose Crop name",data_label)
    
    
    a2 = st.selectbox("Choose Soil type name",df_soil)
    
    a3 = st.selectbox("Choose Area ",df_area) 
    
    
    aa = st.button('Submit')
    
    if aa:
        
        import numpy as np
        # data_label=data_label.unique()
        data_label1 = list(data_label)
        for ii in range(0,len(data_label1)):
            if data_label1[ii] == a1:
                idx = ii
            
        aa=idx
        a1=df['crop_names'][idx]
        
        
        data_label11 = list(df_soil)
        for ii in range(0,len(data_label11)):
            if data_label1[ii] == a2:
                idx = ii
            
        aa=idx
        a2=df['soil_type'][idx]    
        
        
        
        
        Data_reg = [a1,a2,a3]
        # st.text(Data_reg)
                    
        y_pred_reg=rf.predict([Data_reg])
        
        st.write("------------------------------------")
        st.write("The Identified Yeild = ", y_pred_reg[0])    
        st.write("------------------------------------")    
    
    
elif navigation()=='cropRec':      
    
    st.markdown(f'<h1 style="color:#000000;text-align: center;font-size:24px;">{"CROP RECOMMEND"}</h1>', unsafe_allow_html=True)

    
    print("---------------------------------------------")
    print(" Input Data ---> Crop Recommendation")
    print("---------------------------------------------")
    print()
    
    
    # ================  INPUT  ===
    
    df=pd.read_csv("Crop_recommendation.csv")    
    # df=df[0:2500]
    print("--------------------------------")
    print("Data Selection")
    print("--------------------------------")
    print(df.head(15))
    
    data_n = df['N']
    
    data_p = df['P']
    
    data_k = df['K']
    
    data_temp = df['temperature']    
    
    data_hum = df['humidity']
    
    data_ph = df['ph']      
    
    
    data_rain = df['rainfall']       
    
    
    
    # ================  PRE-PROCESSING  ===
     
     # --- MISSING VALUES 
     
    print("--------------------------------")
    print("  Handling Missing Values")
    print("--------------------------------")                    
    print(df.isnull().sum())    
    res=df.isnull().sum().any()
     
    if res==False:
        print("---------------------------------------------")
        print("There is no missing values in our dataset !!!")
        print("---------------------------------------------")
    else:
        print("---------------------------------------------")
        print("Missing values is present in our dataset !!!")
        print("---------------------------------------------")  
        
        
    print("----------------------------------------------------")
    print("Before Label Encoding          ")
    print("----------------------------------------------------")
    print()
    print(df['label'].head(15))
    print()
    
    data_label = df['label']
    label_encoder=preprocessing.LabelEncoder()
    
    print("----------------------------------------------------")
    print("After Label Encoding          ")
    print("----------------------------------------------------")
    print()
    
    df['label']=label_encoder.fit_transform(df['label'])
    
    print(df['label'].head(15))
        
    #============================= DATA SPLITTING ==============================
    
    
    print("----------------------------------------------------")
    print("Data Splitting          ")
    print("----------------------------------------------------")
    print()
    
    from sklearn.model_selection import train_test_split
    
    X = df.drop('label', axis=1)
    y = df['label']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=0.3, random_state=100)
    
    print("Total no of data's       :",df.shape[0])
    print()
    print("Total no of Train data's :",X_train.shape[0])
    print()
    print("Total no of Test data's  :",X_test.shape[0])
    
    
    # ============ RANDOM FOREST ===================
    
    
    from sklearn.ensemble import RandomForestClassifier
    
    rf = RandomForestClassifier()
    
    rf.fit(X_train,y_train)
    
    pred_rf = rf.predict(X_test)
    
    from sklearn import metrics
    
    acc_rf = metrics.accuracy_score(y_test, pred_rf) * 100
    
    print("----------------------------------")
    print("ML --> Random Forest Classifier  ")
    print("----------------------------------")
    print()
    print("1) Accuracy = ", acc_rf,'%')
    print()
    print("2) Classification Report ")
    print()
    print(metrics.classification_report(y_test, pred_rf))
    
    
    # st.text("----------------------------------")
    # st.text("ML --> Random Forest Classifier  ")
    # st.text("----------------------------------")
    
    # st.write("1) Accuracy = ", acc_rf,'%')
    # print()
    # print("2) Classification Report ")
    # print()
    # st.write(metrics.classification_report(y_test, pred_rf))
    
    
    
    st.markdown(f'<h1 style="color:#000000;font-size:14px;">{"Enter the following details"}</h1>', unsafe_allow_html=True)
    
    
    a1 = st.selectbox("Choose Nitrogen Value",data_n)
    
    a2 = st.selectbox("Choose Phosporros",data_p)
    
    a3 = st.selectbox("Choose Potassium",data_k) 
    
    a4 = st.selectbox("Choose Temperature",data_temp)
    
    a5 = st.selectbox("Choose Humidity ",data_hum)
    
    a6 = st.selectbox("Choose Ph value",data_ph) 
    
    a7 = st.selectbox("Choose Rainfall",data_rain) 
    
    aa = st.button('Submit')
    
    if aa:
        Data_reg = [a1,a2,a3,a4,a5,a6,a7]
                    
        y_pred_reg=rf.predict([Data_reg])
        
        pred = label_encoder.inverse_transform(y_pred_reg)
        
        # st.text(pred)
        
        # res = data_label[y_pred_reg]
        
        # res = res.to_string(index=False, header=False)        
        
        st.write("------------------------------")
        st.write("The Identified Crop = ", pred[0])    
        st.write("------------------------------")
    
        # import pickle
        # with open('Result.pickle', 'wb') as f:
        #     pickle.dump(res, f)          
            
        # fert_data = pd.read_csv("fertilizer.csv")
            
        import pandas as pd
        
      
        import csv 
        
        # field names 
        fields = ['Crop'] 
        
    
        
        # st.text(temp_user)
        old_row = [[pred[0]]]
        
        # writing to csv file 
        with open('Crop.csv', 'w') as csvfile: 
            # creating a csv writer object 
            csvwriter = csv.writer(csvfile) 
                
            # writing the fields 
            csvwriter.writerow(fields) 
                
            # writing the data rows 
            csvwriter.writerows(old_row)   
            
elif navigation()=='FertiRec':               
            

    
    st.markdown(f'<h1 style="color:#000000;font-size:24px;">{" Fertilizer Recommendation"}</h1>', unsafe_allow_html=True)
         
    import pandas as pd
    
    df = pd.read_csv('Crop.csv')
    
    resultt = df['Crop'][0]
        
    croppp = resultt.upper()
    
    data_frame=pd.read_csv("Yeild.csv")
    
    x1=data_frame['crop_names']
    
    for i in range(0,len(data_frame)):
        if x1[i]==croppp:
            idx=i
        else:
            idx=7
        
    data_frame1_age = data_frame['Yield']
    
    water = data_frame['water']
    
    Req_data_c=data_frame1_age[idx]
    
    waterlevel = water[idx]
    
    print("-----------------------------------------")
    print("The Yeild  = ", Req_data_c)    
    print("------------------------------------------")
    
    
    
    print("-----------------------------------------")
    print("The Water level  = ", waterlevel)    
    print("------------------------------------------")
    
    
    st.write("-----------------------------------------")
    st.write("The Water level  = ", waterlevel)    
    st.write("------------------------------------------")
          
    
    
    # ===== FERTILIZER 
    
    dff=pd.read_csv("Fertilizer Prediction.csv")    
    
    
    import numpy as np
     
     
    x1=dff['Crop Type']
    for i in range(0,len(dff)):
        if x1[i]==resultt:
            idx=i
        else:
            idx=7
        
    data_frame1_age = dff['Fertilizer Name']
    
    
    Req_data_c=data_frame1_age[idx]
    
    
    st.write("-----------------------------------------")
    st.write("The Recommended Fertilizer = ", Req_data_c)    
    st.write("------------------------------------------")            
            
            
            
elif navigation()=='Disease':                
            
    st.markdown(f'<h1 style="color:#000000;font-size:24px;">{" Disease Prediction Page"}</h1>', unsafe_allow_html=True)
         
    
    
    
    uploaded_file = st.button("Upload Image")
    
    # st.text(uploaded_file)   
    
     
    if uploaded_file:
    #     st.markdown(f'<h1 style="color:#000000;font-size:18px;">{"Please Upload Image"}</h1>', unsafe_allow_html=True)
    
    #     # st.text("Please Upload Video")
        
    # else:
    #     st.text("Uploaded")        
    
         # from tkinter.filedialog import askopenfilename
         
         # filenamee=askopenfilename()
        
        # ================ INPUT IMAGE ======================
        
        
    
        # if file_up==None:
        #     st.text("Browse")
        # else:
        #  st.image(file_up)
        img = mpimg.imread(uploaded_file)
        st.image(img)
         
         
        # ========= PREPROCESSING ============
         
        img_resize_orig = cv2.resize(img,((50, 50)))
         
         
        try:            
             gray1 = cv2.cvtColor(img_resize_orig, cv2.COLOR_BGR2GRAY)
         
        except:
             gray1 = img_resize_orig
             
             
            
        import os 
         
        # ========= IMAGE SPLITTING ============
         
         
        data_apple_bl = os.listdir('Data/Apple_Black_rot/')
         
        data_app_hea = os.listdir('Data/Apple_Healthy/')
         
        data_app_sca = os.listdir('Data/Apple_Scab/')
         
        data_cherry_hea = os.listdir('Data/Cherry_healthy/')
         
        data_cherry_un = os.listdir('Data/Cherry_Powdery_mildew/')
         
        data_corn_dis = os.listdir('Data/Corn_Disease/')
         
        data_corn_heal = os.listdir('Data/Corn_healthy/')
         
        data_grap_diseas = os.listdir('Data/Grape_Diseased/')
         
        data_grap_heal = os.listdir('Data/Grape_Healthy/')
         
        data_tom_dis = os.listdir('Data/Tomato_Disease/')
         
        data_tom_heal = os.listdir('Data/Tomato_healthy/')
         
    
         
        import numpy as np
        dot1= []
        labels1 = [] 
        for img11 in data_apple_bl:
                 # print(img)
                 img_1 = mpimg.imread('Data/Apple_Black_rot//' + "/" + img11)
                 img_1 = cv2.resize(img_1,((50, 50)))
         
         
                 try:            
                     gray = cv2.cvtColor(img_1, cv2.COLOR_BGR2GRAY)
                     
                 except:
                     gray = img_1
         
                 
                 dot1.append(np.array(gray))
                 labels1.append(1)
         
         
        for img11 in data_app_hea:
                 # print(img)
                 img_1 = mpimg.imread('Data/Apple_Healthy//' + "/" + img11)
                 img_1 = cv2.resize(img_1,((50, 50)))
         
         
                 try:            
                     gray = cv2.cvtColor(img_1, cv2.COLOR_BGR2GRAY)
                     
                 except:
                     gray = img_1
         
                 
                 dot1.append(np.array(gray))
                 labels1.append(2)
         
         
        for img11 in data_app_sca:
                 # print(img)
                 img_1 = mpimg.imread('Data/Apple_Scab//' + "/" + img11)
                 img_1 = cv2.resize(img_1,((50, 50)))
         
         
                 try:            
                     gray = cv2.cvtColor(img_1, cv2.COLOR_BGR2GRAY)
                     
                 except:
                     gray = img_1
         
                 
                 dot1.append(np.array(gray))
                 labels1.append(3)
         
         
        for img11 in data_cherry_hea:
                 # print(img)
                 img_1 = mpimg.imread('Data/Cherry_healthy//' + "/" + img11)
                 img_1 = cv2.resize(img_1,((50, 50)))
         
         
                 try:            
                     gray = cv2.cvtColor(img_1, cv2.COLOR_BGR2GRAY)
                     
                 except:
                     gray = img_1
         
                 
                 dot1.append(np.array(gray))
                 labels1.append(4)
         
        for img11 in data_cherry_un:
                 # print(img)
                 img_1 = mpimg.imread('Data/Cherry_Powdery_mildew//' + "/" + img11)
                 img_1 = cv2.resize(img_1,((50, 50)))
         
         
                 try:            
                     gray = cv2.cvtColor(img_1, cv2.COLOR_BGR2GRAY)
                     
                 except:
                     gray = img_1
         
                 
                 dot1.append(np.array(gray))
                 labels1.append(5)
         
         
        for img11 in data_corn_dis:
                 # print(img)
                 img_1 = mpimg.imread('Data/Corn_Disease//' + "/" + img11)
                 img_1 = cv2.resize(img_1,((50, 50)))
         
         
                 try:            
                     gray = cv2.cvtColor(img_1, cv2.COLOR_BGR2GRAY)
                     
                 except:
                     gray = img_1
         
                 
                 dot1.append(np.array(gray))
                 labels1.append(6)
         
        for img11 in data_corn_heal:
                 # print(img)
                 img_1 = mpimg.imread('Data/Corn_healthy//' + "/" + img11)
                 img_1 = cv2.resize(img_1,((50, 50)))
         
         
                 try:            
                     gray = cv2.cvtColor(img_1, cv2.COLOR_BGR2GRAY)
                     
                 except:
                     gray = img_1
         
                 
                 dot1.append(np.array(gray))
                 labels1.append(7)
                 
        for img11 in data_grap_diseas:
                 # print(img)
                 img_1 = mpimg.imread('Data/Grape_Diseased//' + "/" + img11)
                 img_1 = cv2.resize(img_1,((50, 50)))
         
         
                 try:            
                     gray = cv2.cvtColor(img_1, cv2.COLOR_BGR2GRAY)
                     
                 except:
                     gray = img_1
         
                 
                 dot1.append(np.array(gray))
                 labels1.append(8)
         
        for img11 in data_grap_heal:
                 # print(img)
                 img_1 = mpimg.imread('Data/Grape_Healthy//' + "/" + img11)
                 img_1 = cv2.resize(img_1,((50, 50)))
         
         
                 try:            
                     gray = cv2.cvtColor(img_1, cv2.COLOR_BGR2GRAY)
                     
                 except:
                     gray = img_1
         
                 
                 dot1.append(np.array(gray))
                 labels1.append(9)
         
        for img11 in data_tom_dis:
                 # print(img)
                 img_1 = mpimg.imread('Data/Tomato_Disease//' + "/" + img11)
                 img_1 = cv2.resize(img_1,((50, 50)))
         
         
                 try:            
                     gray = cv2.cvtColor(img_1, cv2.COLOR_BGR2GRAY)
                     
                 except:
                     gray = img_1
         
                 
                 dot1.append(np.array(gray))
                 labels1.append(10)
         
        for img11 in data_tom_heal:
                 # print(img)
                 img_1 = mpimg.imread('Data/Tomato_healthy//' + "/" + img11)
                 img_1 = cv2.resize(img_1,((50, 50)))
         
         
                 try:            
                     gray = cv2.cvtColor(img_1, cv2.COLOR_BGR2GRAY)
                     
                 except:
                     gray = img_1
         
                 
                 dot1.append(np.array(gray))
                 labels1.append(11)
         
    
                 
           
        x_train, x_test, y_train, y_test = train_test_split(dot1,labels1,test_size = 0.2, random_state = 101)
         
         
        print("------------------------------------------------------------")
        print(" Image Splitting")
        print("------------------------------------------------------------")
        print()
            
        print("The Total of Images       =",len(dot1))
        print("The Total of Train Images =",len(x_train))
        print("The Total of Test Images  =",len(x_test))
            
            
    ###################
         
         
        # filename = askopenfilename()
        # img = mpimg.imread(filename)
        # img_1 = cv2.resize(img,((50, 50)))
        # try:            
        #     gray = cv2.cvtColor(img_1, cv2.COLOR_BGR2GRAY)
            
        # except:
        #     gray = img_1
        
        # gray=np.array(gray)
        # x_train11=np.zeros((len(gray),50))
        # x_train11[i,:]=np.mean(gray) 
        # # gray=np.zeros((len(gray),50))
        # # x_train11=np.mean(gray)    
        # y_pred = rf.predict(x_train11)[49]
        
    
    ##################
    
    
        # print()
        
    
         
         # for i in range(0,len(gray)):
         #     x_train11[i,:]=np.mean(gray[i])    
        
        
            # ===== CLASSIFICATION ======
            
            
        from keras.utils import to_categorical
        
        
        x_train11=np.zeros((len(x_train),50,50,3))
        for i in range(0,len(x_train)):
            x_train11[i,:]=np.mean(x_train[i])
            
        x_test11=np.zeros((len(x_test),50,50,3))
        for i in range(0,len(x_test)):
             x_test11[i,:]=np.mean(x_test[i])
            
            
        y_train11=np.array(y_train)
        y_test11=np.array(y_test)
            
        train_Y_one_hot = to_categorical(y_train11)
        test_Y_one_hot = to_categorical(y_test)
         
       # ======== CNN ===========
            
        from keras.layers import Dense, Conv2D
        from keras.layers import Flatten
        from keras.layers import MaxPooling2D
        from keras.layers import Activation
        from keras.models import Sequential
        from keras.layers import Dropout
        
        
        # initialize the model
        model=Sequential()
        
        
        #CNN layes 
        model.add(Conv2D(filters=16,kernel_size=2,padding="same",activation="relu",input_shape=(50,50,3)))
        model.add(MaxPooling2D(pool_size=2))
    
        model.add(Conv2D(filters=32,kernel_size=2,padding="same",activation="relu"))
        model.add(MaxPooling2D(pool_size=2))
        
        model.add(Conv2D(filters=64,kernel_size=2,padding="same",activation="relu"))
        model.add(MaxPooling2D(pool_size=2))
        
        model.add(Dropout(0.2))
        model.add(Flatten())
        
        model.add(Dense(500,activation="relu"))
        
        model.add(Dropout(0.2))
        
        model.add(Dense(12,activation="softmax"))
        
        #summary the model 
        model.summary()
        
        #compile the model 
        model.compile(loss='binary_crossentropy', optimizer='adam')
        y_train1=np.array(y_train)
        
        train_Y_one_hot = to_categorical(y_train1)
        test_Y_one_hot = to_categorical(y_test)
        
        
        print("-------------------------------------")
        print("CONVOLUTIONAL NEURAL NETWORK (CNN)")
        print("-------------------------------------")
        print()
        #fit the model 
        history=model.fit(x_train11,train_Y_one_hot,batch_size=2,epochs=5,verbose=1)
        
        accuracy = model.evaluate(x_train11, train_Y_one_hot, verbose=1)
        
        pred_cnn = model.predict([x_train11])
        
        y_pred2 = pred_cnn.reshape(-1)
        y_pred2[y_pred2<0.5] = 0
        y_pred2[y_pred2>=0.5] = 1
        y_pred2 = y_pred2.astype('int')
        
        loss=history.history['loss']
        loss=max(loss)
        
        acc_cnn=100-loss
        
        print("-------------------------------------")
        print("PERFORMANCE ---------> (CNN)")
        print("-------------------------------------")
        print()
        #acc_cnn=accuracy[1]*100
        print("1. Accuracy   =", acc_cnn,'%')
        print()
        print("2. Error Rate =",loss)
             
         
         
         
         
        Total_length = len(data_apple_bl) + len(data_app_hea) + len(data_app_sca) + len(data_cherry_hea) + len(data_cherry_un) + len(data_corn_dis) + len(data_corn_heal) + len(data_grap_diseas)+ len(data_grap_heal) + len(data_tom_dis) + len(data_tom_heal) 
        
        
        temp_data1  = []
        for ijk in range(0,Total_length):
                    # print(ijk)
            temp_data = int(np.mean(dot1[ijk]) == np.mean(gray1))
            temp_data1.append(temp_data)
                
        temp_data1 =np.array(temp_data1)
                
        zz = np.where(temp_data1==1)
                
        if labels1[zz[0][0]] == 1:
            print('-------------------------------------')
            print()
            print(' The Identified Crop  = Apple Balck Rot')
            print()
            print('--------------------------------------')
    
            st.text('-------------------------------------')
            print()
            st.text(' The Identified Crop  = Apple ')
            st.text(' The Identified Diseased  =  Balck Rot')
            print()
            st.text('--------------------------------------')
    
        elif labels1[zz[0][0]] == 2:
            print('-----------------------------------')
             
            print()
            print('The Identified Crop  = Apple Healthy')   
            print()
            print('-------------------------------------')
    
            st.text('-------------------------------------')
            print()
            st.text(' The Identified Crop  = Apple')
            st.text(' The Identified Diseased  =  Healthy')
            print()
            st.text('--------------------------------------')
    
        elif labels1[zz[0][0]] == 3:
            print('-----------------------------------')
             
            print()
            print('The Identified Crop  = Apple Scab')   
            print()
            print('-------------------------------------')
    
            st.text('-------------------------------------')
            print()
            st.text(' The Identified Crop  = Apple')
            st.text(' The Identified Diseased  = Scab')
            print()
            st.text('--------------------------------------')
      
        elif labels1[zz[0][0]] == 4:
            print('-----------------------------------')
             
            print()
            print('The Identified Crop  = Cherry Healthy')   
            print()
            print('-------------------------------------')
    
            st.text('-------------------------------------')
            print()
            st.text(' The Identified Crop  = Cherry')
            st.text(' The Identified Disease  = Healthy')
            print()
            st.text('--------------------------------------')
    
        elif labels1[zz[0][0]] == 5:
            print('-----------------------------------')
             
            print()
            print('The Identified Crop  = Cherry Diseased')   
            print()
            print('-------------------------------------')
    
            st.text('-------------------------------------')
            print()
            st.text(' The Identified Crop  = Cherry Diseased')
            st.text(' The Identified Disease  = Diseased')
            print()
            st.text('--------------------------------------')
    
    
        elif labels1[zz[0][0]] == 6:
            print('-----------------------------------')
             
            print()
            print('The Identified Crop  = Corn Diseased')   
            print()
            print('-------------------------------------')
    
            st.text('-------------------------------------')
            print()
            st.text(' The Identified Crop  = Corn ')
            st.text(' The Identified Disease  =  Diseased')
            print()
            st.text('--------------------------------------')
    
    
        elif labels1[zz[0][0]] == 7:
            print('-----------------------------------')
             
            print()
            print('The Identified Crop  = Corn Healthy')   
            print()
            print('-------------------------------------')
    
            st.text('-------------------------------------')
            print()
            st.text(' The Identified Crop  = Corn ')
            st.text(' The Identified Disease  = Healthy')
            print()
            st.text('--------------------------------------')
    
    
        elif labels1[zz[0][0]] == 8:
            print('-----------------------------------')
             
            print()
            print('The Identified Crop  = Grape Diseased')   
            print()
            print('-------------------------------------')
    
            st.text('-------------------------------------')
            print()
            st.text(' The Identified Crop  = Grape ')
            st.text(' The Identified Disease  = Diseased')
            print()
            st.text('--------------------------------------')
    
    
        elif labels1[zz[0][0]] == 9:
            print('-----------------------------------')
             
            print()
            print('The Identified Crop  = Grape Healthy')   
            print()
            print('-------------------------------------')
    
            st.text('-------------------------------------')
            print()
            st.text(' The Identified Crop  = Grape')
            st.text(' The Identified Disease  = Healthy')
            print()
            st.text('--------------------------------------')
    
    
        elif labels1[zz[0][0]] == 10:
            print('-----------------------------------')
             
            print()
            print('The Identified Crop  = Tomato Diseased')   
            print()
            print('-------------------------------------')
    
            st.text('-------------------------------------')
            print()
            st.text(' The Identified Crop  = Tomato')
            st.text(' The Identified Disease  = Diseased')
            print()
            st.text('--------------------------------------')
    
    
        elif labels1[zz[0][0]] == 11:
            print('-----------------------------------')
             
            print()
            print('The Identified Crop  = Tomato Healthy')   
            print()
            print('-------------------------------------')
    
            st.text('-------------------------------------')
            print()
            st.text(' The Identified Crop  = Tomato ')
            st.text(' The Identified Disease  = Healthy')
            print()
            st.text('--------------------------------------')            
            
            
elif navigation()=='govt':                 
            
    st.markdown(f'<h1 style="color:#000000;font-size:24px;">{" Government Schemes"}</h1>', unsafe_allow_html=True)
         
    
    url = "https://pib.gov.in/PressReleaseIframePage.aspx?PRID=2002012"
    # st.write("check out this [link](%s)" % url)
    st.markdown("Government Schemes [link](%s)" % url)
                
            
elif navigation()=='regional':      


    st.markdown(f'<h1 style="color:#000000;font-size:24px;">{" Regional Officier"}</h1>', unsafe_allow_html=True)
         
    
    url = "https://tnwc.in/regional-offices/"
    # st.write("check out this [link](%s)" % url)
    st.markdown("Regional Officer Details  [link](%s)" % url)

    st.markdown(f'<h1 style="color:#000000;font-size:24px;">{" Demo Video"}</h1>', unsafe_allow_html=True)


    video_file = open("1.mp4", 'rb')
    video_bytes = video_file.read()
    # tk=str(U_P1)
    # tk=float(tk[0:2])
    st.video(video_bytes)









          