import streamlit as st
from pages import home, get_score, portfolio, analysis, about_us

# Page navigation
PAGES = {
    "Home": home,
    "Get Score": get_score,
    "Portfolio": portfolio,
    "Analysis": analysis,
    # "Wishlist": wishlist,
    # "Rebalancing": rebalancing,
    "About Us": about_us,
}

def main():
    st.sidebar.title("Navigation")
    selection = st.sidebar.radio("Go to", list(PAGES.keys()))

    page = PAGES[selection]
    page()  # Call the page function directly

if __name__ == "__main__":
    main()
