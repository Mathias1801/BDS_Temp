# Link to huggingface display: https://mathiasbds-assignment-2-1-semester.hf.space
# Import necessary libraries
import streamlit as st
import pandas as pd
import altair as alt
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import pycountry

# Function to load the dataset
@st.cache_data  # Cache the function to enhance performance
def load_data():
    # Define the file path
    file_path = 'Data/GEM_data_2022.csv'
    
    # Load the CSV file into a pandas dataframe
    data = pd.read_csv(file_path)

    # Isolating desired columns and cleaning the data
    df = data[['age', 'gender', 'gemeduc', 'fearfaill', 'suskilll', 'country_name']]
    df = df.dropna()

    #Creating age categories
    Labels = ['18-24', '25-34', '35-44', '45-54', '55-64', '65+']
    bins = [18, 24, 34, 44, 54, 64, 100]
    df['age_cat'] = pd.cut(df.age, bins=bins, labels=Labels)
    age_cat = df['age_cat']

    #Renaming gender
    df['gender'] = df['gender'].replace({1: 'Male', 2: 'Female'})
    gender = df['gender']

    return df
# Load the data using the defined function
df = load_data()

# Set the app title and sidebar header
st.title("Entrepeneurs around the world - Dashboard")
st.sidebar.header("Filters")

# Global Entrepeneurship Monitor Dashboard
st.markdown("""
            Welcome to the Fear of Failure Insights Dashboard. Against the backdrop of rising concerns about fear of failure, the GEM dataset sheds light on its impact across various domains. Through data analytics, this dashboard explores the underlying factors contributing to fear of failure and offers strategies to help mitigate its effects. Dive deep into the insights to better understand how fear of failure influences behavior and decision-making, and discover ways to foster resilience and confidence.
""")

st.image("https://stetsonbalfour.com/wp-content/uploads/2023/04/Entrepreneur.jpg", caption= "📈 Entrepreneur looking at chalk board of success 📉", use_column_width=True)

# Function to get ISO Alpha-3 code
def get_iso_alpha_3(country_name):
    try:
        return pycountry.countries.lookup(country_name).alpha_3
    except LookupError:
        return None

# Process the data: Compute the frequency of occurrences by country
country_counts = df['country_name'].value_counts().reset_index()
country_counts.columns = ['Country', 'Count']

# Convert country names to ISO Alpha-3 codes
country_counts['iso_alpha_3'] = country_counts['Country'].apply(get_iso_alpha_3)

# Drop rows with None values in 'iso_alpha_3' column
country_counts = country_counts.dropna(subset=['iso_alpha_3'])

# Create the map visualization
fig = px.choropleth(country_counts, locations='iso_alpha_3', locationmode='ISO-3',
                    color='Count', hover_name='Country',
                    color_continuous_scale='Blues', title="Frequency of GEM Data by Country")

# Display the map in Streamlit
st.plotly_chart(fig)

with st.expander("**Objective**"):
                 st.markdown("""
At the heart of this dashboard is the mission to visually decode data, equipping researchers with insights to address key questions about fear of failure in entrepreneurship, focusing on gender, education, sufficient skills, and age:
Which gender or age groups are more likely to experience fear of failure in entrepreneurship?
- The dashboard will highlight trends in how fear of failure affects different genders and age groups, revealing disparities that may require attention.

How does education level impact fear of failure, and does having sufficient entrepreneurial skills reduce this fear?
- By examining the relationship between education and self-perceived skills, the data will reveal whether higher education or skill confidence reduces fear of failure, especially across different gender and age categories.

Based on these trends, what strategies can encourage entrepreneurship by addressing fear of failure in specific demographic groups?
- Insights into how gender, age, education, and skills correlate with fear of failure will help tailor support systems, such as skill-building workshops or mentoring programs, to reduce barriers to entrepreneurship.

"""
)
                             
# Tutorial Expander
with st.expander("How to Use the Dashboard"):
    st.markdown("""
    1. Use the **sidebar filters** to highlight and isolate specific parts of the data.
    2. **Visualization** type and paterns can be found in the drop down menu.
    3. Further down the page you will find **recommendations and deriviation** based on the data-analysis
    """)

# Filter Expander
with st.expander("Explainer of filters"):
    st.markdown("""
    - **Age Group:** This filter consists of ranges of ages from 18 to 65 years or more.
    - **Gender:** This filter consists of male and female filtration options.
    - **Degree of Completed Education Range:** This filter consists of numerical values representing the degree of a respondents completed degree of education. The higher the value, the larger the completed educational level.
    - **Select degree of Sufficient Skill Levels:** This filter consists of the respondents selfevaluated degree of sufficient skill to start a business.
    """)

# Sidebar filter: Age Group 'age_cat'
selected_age_group = st.sidebar.multiselect("Select Age Groups", df['age_cat'].unique().tolist(), default=df['age_cat'].unique().tolist())
if not selected_age_group:
    st.warning("Please select an age group from the sidebar ⚠️")
    st.stop()
filtered_df = df[df['age_cat'].isin(selected_age_group)]

# Sidebar filter: Gender 'gender'
genders = df['gender'].unique().tolist()
selected_gender = st.sidebar.multiselect("Select Gender", genders, default=genders)
if not selected_gender:
    st.warning("Please select a Gender from the sidebar ⚠️")
    st.stop()
filtered_df = filtered_df[filtered_df['gender'].isin(selected_gender)]

# Sidebar filter: Degree of completed education 'gemeduc'
min_gemeduc = int(df['gemeduc'].min())
max_gemeduc = int(df['gemeduc'].max())
gemduc_range = st.sidebar.slider("Select Degree of Completed Education Range", min_gemeduc, max_gemeduc, (min_gemeduc, max_gemeduc))
filtered_df = filtered_df[(filtered_df['gemeduc'] >= gemduc_range[0]) & (filtered_df['gemeduc'] <= gemduc_range[1])]

# Sidebar filter: Sufficient skill 'suskilll'
suskilll_levels = sorted(df['suskilll'].unique().tolist())
selected_suskilll = st.sidebar.multiselect("Select degree of Sufficient Skill Levels", suskilll_levels, default=suskilll_levels)
if not selected_suskilll:
    st.warning("Please select a Sufficient Skill level from the sidebar ⚠️")
    st.stop()
filtered_df = filtered_df[filtered_df['suskilll'].isin(selected_suskilll)]

# Displaying the Fear of Failure Analysis header
st.header("Fear of Failure Analysis")

# Dropdown to select the type of visualization
visualization_option = st.selectbox(
    "Select Visualization", 
    ["Fear of failure by Age Group", 
     "KDE Plot: Degree of Completed Education by Fear of Failure", 
     "Fear of Failure by Sufficient Degree of Skill", 
     "Fear of Failure Distribution by Gender", 
     "Correlation matrix of data set"]
)

# Visualizations based on user selection
if visualization_option == "Fear of failure by Age Group":
    # Bar chart for Fear of failure by Age Group
    chart = alt.Chart(filtered_df).mark_bar().encode(
        x='age_cat',
        y='count()',
        color='fearfaill'
    ).properties(
        title='Fear of Failure Rate by Age Group'
    )
    st.altair_chart(chart, use_container_width=True)

elif visualization_option == "KDE Plot: Degree of Completed Education by Fear of Failure":
    # KDE plot for Degree of Completed Education by Fear of Failure
    plt.figure(figsize=(10, 6))
    sns.kdeplot(data=filtered_df, x='gemeduc', hue='fearfaill', fill=True, palette='Set2')
    plt.xlabel('Education level')
    plt.ylabel('Density')
    plt.title('KDE Plot of Degree of Completed Education by Fear of Failure')
    st.pyplot(plt)

elif visualization_option == "Fear of Failure by Sufficient Degree of Skill":
    # Bar chart for fear of failure by sufficent skill eval
    chart = alt.Chart(filtered_df).mark_bar().encode(
        y='suskilll',
        x='count()',
        color='fearfaill'
    ).properties(
        title='Fear of failure by degree of self-evaluated sufficient skill'
    )
    st.altair_chart(chart, use_container_width=True)

elif visualization_option == "Fear of Failure Distribution by Gender":
    # Boxplots for Fear of Failure Distribution by Gender
    fig, ax = plt.subplots(1, 2, figsize=(15, 7))
    sns.boxplot(x="gender", y="fearfaill", data=filtered_df, ax=ax[0], hue="gender", palette='Set2', legend=False)
    ax[0].set_title('Fear of Failure Distribution by Gender')
    ax[0].set_xlabel('Gender')
    ax[0].set_ylabel('Fear of Failure')
    
    plt.tight_layout()
    st.pyplot(fig)

elif visualization_option == "Correlation matrix of data set":
    #Heatmap for the dataset
    df_corr = df.drop(['age_cat', 'gender', 'country_name'],axis=1).corr()
    fig, ax = plt.subplots()
    sns.heatmap(df_corr, annot=True, cmap='coolwarm', ax=ax)
    st.pyplot(fig)

# Display dataset overview
st.header("Data set Overview")
st.dataframe(df.describe(), width=800, height=320)

# Insights from Visualization Section Expander
with st.expander("Insights from Visualization"):
    st.markdown("""
    1. Age Groups & Fear of Failure - The 'Fear of Failure by Age Group' plot highlights which age brackets are more likely to experience fear of failure when considering entrepreneurship. This can inform policies to support older or younger potential entrepreneurs.
    2. Education's Impact - The 'KDE Plot: Education Level by Fear of Failure' visualizes how educational attainment correlates with fear of failure. It provides insights into whether individuals with higher education tend to have lower fear or if additional training is needed at any level.
    3. Skills & Fear of Failure - The 'Fear of Failure by Sufficient Skills' plot shows how self-perceived entrepreneurial skills influence fear. This helps identify whether individuals who feel they have sufficient skills are less likely to fear failure.
    4. Gender & Fear of Failure - The pie chart for 'Fear of Failure Distribution by Gender' reveals any gender-based trends in fear of failure, helping identify if one gender may require more targeted interventions or encouragement to pursue entrepreneurship.
    5. Correlation matrix - Through this chart we identify sufficent skill as the column having the highest correlation value relative to Fear of Failure potentially resulting in a causality realationship.
    """)

# Recommendations Expander
with st.expander("Recommendations for Action"):
    st.markdown("""
    - 🎯 Targeted Support Programs: Develop mentorship and training initiatives specifically for age groups and genders showing higher levels of fear of failure, helping them build confidence and reduce barriers to entry into entrepreneurship.
    - 📚 Skill-Building Workshops: Offer skill development programs for individuals, especially those who report insufficient entrepreneurial skills, to lower fear of failure. Tailor these programs by age and education levels to maximize impact.
    - 🌐 Education-Based Interventions: Encourage educational programs that bridge the gap between formal education and entrepreneurial skills, focusing on age groups and educational levels with higher fear of failure.
    - 👫 Gender-Specific Support: Create entrepreneurial programs aimed at encouraging women or other genders disproportionately impacted by fear of failure, fostering a more inclusive entrepreneurial environment.
    - 👩‍💼 Lifelong Learning Opportunities: Provide ongoing learning opportunities and skill refreshers for older age groups that may feel underprepared for entrepreneurship, helping to build confidence and reduce fear in taking entrepreneurial risks.

    """)

# Feedback system
st.title("Did this app prove useful?")

# Predefined Feedback options
options = ['😇 Yes, it is amazing!', '😃 I found it somewhat useful', '🥹 It lacked some deeper analysis, but works as an overview', '🤯 It was terrible (I am too shy to say it was amazing!)']

# Voting counts (stored in session state)
if 'feedback' not in st.session_state:
    st.session_state['feedback'] = [0] * len(options)

# Feedback mechanism
st.write("### Vote for your favorite option:")
selected_option = st.selectbox('Choose an option:', options)

if st.button('Choose'):
    idx = options.index(selected_option)
    st.session_state['feedback'][idx] += 1
    st.write(f"Thanks for your feedback: {selected_option}!")

# Display voting results
results_df = pd.DataFrame({
    'Option': options,
    'feedback': st.session_state['feedback']
})

st.write("### Live Feedback Results:")
st.write(results_df)

# Bar chart of results
fig = px.bar(results_df, x='Option', y='feedback', title='Feedback Results')
st.plotly_chart(fig)