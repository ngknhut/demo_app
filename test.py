import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
import io
import requests
from sklearn.metrics.pairwise import cosine_similarity
import plotly.express as px
import plotly.graph_objects as go

# Thiết lập cấu hình trang
st.set_page_config(
    page_title="Hệ thống Đề xuất Sản phẩm Shopee",
    page_icon="shopee.png",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Hàm mô phỏng để lấy dữ liệu sản phẩm
def get_sample_products(n=10):
    products = {
        'product_id': [f'P{i:04d}' for i in range(1, n+1)],
        'product_name': [f'Sản phẩm {i}' for i in range(1, n+1)],
        'category': np.random.choice(['Thời trang', 'Điện tử', 'Gia dụng', 'Mỹ phẩm', 'Đồ chơi'], n),
        'price': np.random.randint(100000, 2000000, n),
        'rating': np.random.uniform(3.5, 5.0, n).round(1),
        'image_url': [f'https://picsum.photos/id/{i+20}/200/200' for i in range(n)]
    }
    return pd.DataFrame(products)

# Hàm mô phỏng để lấy dữ liệu khách hàng
def get_sample_customers(n=5):
    customers = {
        'customer_id': [f'C{i:04d}' for i in range(1, n+1)],
        'name': [f'Khách hàng {i}' for i in range(1, n+1)]
    }
    return pd.DataFrame(customers)

# Hàm mô phỏng kết quả đã đạt được khi huấn luyện mô hình
def get_training_results():
    results = {
        'model': ['Content-based', 'Collaborative Filtering', 'Hybrid Model'],
        'precision': [0.72, 0.68, 0.78],
        'recall': [0.65, 0.73, 0.76],
        'f1_score': [0.68, 0.70, 0.77],
        'map': [0.62, 0.67, 0.74],
        'ndcg': [0.69, 0.71, 0.80]
    }
    return pd.DataFrame(results)

# Hàm mô phỏng để tìm sản phẩm tương tự dựa trên nội dung
def find_similar_products(description, products_df, n=10):
    # Trong thực tế, bạn sẽ sử dụng các vector đặc trưng từ mô hình đã huấn luyện
    # Đây chỉ là mô phỏng
    np.random.seed(hash(description) % 10000)
    similarity_scores = np.random.uniform(0.5, 1.0, len(products_df))
    products_df['similarity'] = similarity_scores
    return products_df.sort_values('similarity', ascending=False).head(n)

# Hàm mô phỏng để lấy đề xuất dựa trên ID khách hàng
def get_customer_recommendations(customer_id, products_df, n=10):
    # Trong thực tế, bạn sẽ sử dụng mô hình đã huấn luyện
    # Đây chỉ là mô phỏng
    np.random.seed(hash(customer_id) % 10000)
    recommendation_scores = np.random.uniform(0.5, 1.0, len(products_df))
    products_df['recommendation_score'] = recommendation_scores
    return products_df.sort_values('recommendation_score', ascending=False).head(n)

# Hàm hiển thị sản phẩm dưới dạng thẻ
def display_product_cards(products, score_col=None):
    num_products = len(products)
    cols = st.columns(5)
    
    for i, (_, product) in enumerate(products.iterrows()):
        col_idx = i % 5
        with cols[col_idx]:
            st.image(product['image_url'], width=150)
            st.markdown(f"**{product['product_name']}**")
            st.write(f"Danh mục: {product['category']}")
            st.write(f"Giá: {product['price']:,} đ")
            st.write(f"Đánh giá: {product['rating']}⭐")
            if score_col:
                st.write(f"Độ tương đồng: {product[score_col]:.2f}")
            st.markdown("---")

# Sidebar
st.sidebar.title("Hệ thống Đề xuất Sản phẩm")
st.sidebar.image("shopee_pic_1.jpg", width=250)
st.sidebar.markdown("---")

# Chọn trang trong sidebar
page = st.sidebar.selectbox(
    "Chọn chức năng:",
    ["Kết quả huấn luyện", "Tìm sản phẩm tương tự", "Đề xuất cá nhân hóa"]
)

# Tải dữ liệu mẫu
sample_products = get_sample_products(50)
sample_customers = get_sample_customers()
training_results = get_training_results()

# Trang 1: Kết quả huấn luyện
if page == "Kết quả huấn luyện":
    st.title("Kết quả Huấn luyện Mô hình Đề xuất")
    
    # Hiển thị thông tin tổng quan
    st.subheader("Thông tin tổng quan về Dự án")
    col1, col2, col3 = st.columns(3)
    col1.metric("Tổng số sản phẩm", "10,000+", "")
    col2.metric("Tổng số người dùng", "50,000+", "")
    col3.metric("Tổng số đánh giá", "500,000+", "")
    
    st.markdown("---")
    
    # Hiển thị bảng kết quả
    st.subheader("Đánh giá hiệu suất các mô hình")
    st.table(training_results.style.highlight_max(axis=0, color='lightgreen'))
    
    # Hiển thị biểu đồ so sánh các mô hình
    st.subheader("So sánh hiệu suất các mô hình")
    
    # Biểu đồ cột so sánh các metric
    metrics = ['precision', 'recall', 'f1_score', 'map', 'ndcg']
    fig = go.Figure()
    
    for model in training_results['model']:
        model_results = training_results[training_results['model'] == model]
        fig.add_trace(go.Bar(
            x=metrics,
            y=model_results[metrics].values[0],
            name=model
        ))
    
    fig.update_layout(
        title="So sánh các metric đánh giá",
        xaxis_title="Metric",
        yaxis_title="Giá trị",
        legend_title="Mô hình",
        barmode='group',
        height=500
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Biểu đồ hiển thị sự cải thiện qua các epoch
    st.subheader("Đường cong huấn luyện của mô hình tốt nhất (Hybrid Model)")
    
    # Tạo dữ liệu mô phỏng cho quá trình huấn luyện
    epochs = list(range(1, 21))
    train_loss = [0.8 - 0.03 * e + 0.002 * e**2 for e in range(1, 21)]
    val_loss = [0.85 - 0.025 * e + 0.0015 * e**2 for e in range(1, 21)]
    
    training_df = pd.DataFrame({
        'Epoch': epochs + epochs,
        'Loss': train_loss + val_loss,
        'Type': ['Training'] * 20 + ['Validation'] * 20
    })
    
    fig2 = px.line(
        training_df, 
        x='Epoch', 
        y='Loss', 
        color='Type',
        title='Đường cong học tập qua các epoch',
        markers=True
    )
    
    st.plotly_chart(fig2, use_container_width=True)
    
    # Thêm phần phân phối đề xuất
    st.subheader("Phân phối đề xuất theo danh mục sản phẩm")
    
    # Tạo dữ liệu phân phối đề xuất
    category_dist = {
        'Danh mục': ['Thời trang', 'Điện tử', 'Gia dụng', 'Mỹ phẩm', 'Đồ chơi'],
        'Số lượng đề xuất': [4500, 3200, 1800, 900, 600]
    }
    
    fig3 = px.pie(
        values=category_dist['Số lượng đề xuất'],
        names=category_dist['Danh mục'],
        title='Tỷ lệ đề xuất theo danh mục'
    )
    
    st.plotly_chart(fig3, use_container_width=True)

# Trang 2: Tìm sản phẩm tương tự
elif page == "Tìm sản phẩm tương tự":
    st.title("Tìm Sản phẩm Tương tự (Content-based)")
    
    # Nhập mô tả sản phẩm
    product_description = st.text_area(
        "Nhập mô tả sản phẩm hoặc từ khóa cần tìm:",
        height=100,
        placeholder="Ví dụ: Áo thun cotton nam phong cách thể thao, màu xanh navy, chất vải mềm thoáng khí..."
    )
    
    # Bộ lọc bổ sung
    with st.expander("Bộ lọc nâng cao"):
        col1, col2 = st.columns(2)
        with col1:
            categories = st.multiselect(
                "Danh mục sản phẩm:",
                ["Tất cả"] + list(sample_products['category'].unique()),
                default="Tất cả"
            )
        with col2:
            price_range = st.slider(
                "Khoảng giá (đồng):",
                0, 2000000, (0, 2000000)
            )
    
    # Nút tìm kiếm
    search_btn = st.button("Tìm kiếm", type="primary")
    
    if search_btn and product_description:
        # Áp dụng bộ lọc 
        filtered_products = sample_products.copy()
        
        if "Tất cả" not in categories:
            filtered_products = filtered_products[filtered_products['category'].isin(categories)]
        
        filtered_products = filtered_products[
            (filtered_products['price'] >= price_range[0]) & 
            (filtered_products['price'] <= price_range[1])
        ]
        
        # Tìm sản phẩm tương tự
        similar_products = find_similar_products(product_description, filtered_products)
        
        # Hiển thị kết quả
        st.subheader(f"Sản phẩm tương tự với mô tả của bạn:")
        st.info(f"Tìm thấy {len(similar_products)} sản phẩm tương tự dựa trên mô tả của bạn.")
        
        # Hiển thị sản phẩm dưới dạng thẻ
        display_product_cards(similar_products, score_col='similarity')
        
        # Hiển thị biểu đồ phân tích
        st.subheader("Phân tích độ tương đồng")
        
        # Biểu đồ phân phối độ tương đồng
        fig = px.bar(
            similar_products,
            x='product_name',
            y='similarity',
            color='category',
            title="Điểm tương đồng của các sản phẩm được đề xuất",
            labels={'product_name': 'Sản phẩm', 'similarity': 'Điểm tương đồng'}
        )
        fig.update_layout(xaxis_tickangle=-45)
        st.plotly_chart(fig, use_container_width=True)
    
    # Hiển thị hướng dẫn nếu chưa nhập mô tả
    if not search_btn or not product_description:
        st.info("👆 Vui lòng nhập mô tả sản phẩm bạn quan tâm để nhận đề xuất sản phẩm tương tự.")
        
        # Hiển thị một số sản phẩm phổ biến để tham khảo
        st.subheader("Một số sản phẩm phổ biến")
        popular_products = sample_products.sample(10)
        display_product_cards(popular_products)

# Trang 3: Đề xuất cá nhân hóa
elif page == "Đề xuất cá nhân hóa":
    st.title("Đề xuất Sản phẩm Cá nhân hóa")
    
    # Tạo tab để chọn phương pháp đề xuất
    tab1, tab2 = st.tabs(["Dựa trên người dùng", "Kết hợp nội dung"])
    
    # Tab 1: Đề xuất dựa trên ID người dùng
    with tab1:
        # Chọn ID khách hàng từ danh sách
        customer_id = st.selectbox(
            "Chọn mã khách hàng:",
            options=sample_customers['customer_id'],
            format_func=lambda x: f"{x} - {sample_customers[sample_customers['customer_id'] == x]['name'].values[0]}"
        )
        
        # Hoặc nhập ID khách hàng mới
        custom_id = st.text_input("Hoặc nhập mã khách hàng khác:")
        
        if custom_id:
            customer_id = custom_id
        
        # Nút tìm kiếm
        rec_btn = st.button("Lấy đề xuất", key="rec_btn1", type="primary")
        
        if rec_btn and customer_id:
            # Lấy đề xuất sản phẩm dựa trên ID khách hàng
            recommended_products = get_customer_recommendations(customer_id, sample_products)
            
            # Hiển thị tiêu đề với ID khách hàng
            st.subheader(f"Sản phẩm đề xuất cho khách hàng {customer_id}:")
            
            # Hiển thị sản phẩm đề xuất
            display_product_cards(recommended_products, score_col='recommendation_score')
            
            # Hiển thị phân tích đề xuất
            st.subheader("Phân tích đề xuất")
            
            # Biểu đồ phân loại đề xuất theo danh mục
            category_counts = recommended_products['category'].value_counts().reset_index()
            category_counts.columns = ['Danh mục', 'Số lượng']
            
            fig = px.pie(
                category_counts,
                values='Số lượng',
                names='Danh mục',
                title='Phân loại đề xuất theo danh mục'
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Biểu đồ điểm đề xuất
            fig2 = px.bar(
                recommended_products,
                x='product_name',
                y='recommendation_score',
                color='category',
                title="Điểm đề xuất cho từng sản phẩm",
                labels={'product_name': 'Sản phẩm', 'recommendation_score': 'Điểm đề xuất'}
            )
            fig2.update_layout(xaxis_tickangle=-45)
            st.plotly_chart(fig2, use_container_width=True)
    
    # Tab 2: Đề xuất kết hợp nội dung và ID người dùng
    with tab2:
        # Chọn ID khách hàng từ danh sách
        customer_id2 = st.selectbox(
            "Chọn mã khách hàng:",
            options=sample_customers['customer_id'],
            format_func=lambda x: f"{x} - {sample_customers[sample_customers['customer_id'] == x]['name'].values[0]}",
            key="customer_select2"
        )
        
        # Nhập mô tả sản phẩm
        product_description2 = st.text_area(
            "Nhập mô tả sản phẩm hoặc từ khóa quan tâm:",
            height=100,
            placeholder="Ví dụ: Áo thun cotton nam phong cách thể thao, màu xanh navy, chất vải mềm thoáng khí...",
            key="desc_area2"
        )
        
        # Điều chỉnh tỷ lệ ảnh hưởng
        hybrid_ratio = st.slider(
            "Tỷ lệ: Dựa trên người dùng - Dựa trên nội dung",
            0.0, 1.0, 0.5, 0.1,
            help="0.0 = Hoàn toàn dựa trên nội dung, 1.0 = Hoàn toàn dựa trên người dùng"
        )
        
        # Nút tìm kiếm
        hybrid_btn = st.button("Lấy đề xuất kết hợp", key="rec_btn2", type="primary")
        
        if hybrid_btn and customer_id2 and product_description2:
            # Lấy đề xuất từ cả hai phương pháp
            user_based = get_customer_recommendations(customer_id2, sample_products)
            content_based = find_similar_products(product_description2, sample_products)
            
            # Kết hợp hai kết quả với tỷ lệ đã chọn
            hybrid_products = sample_products.copy()
            
            # Đảm bảo tất cả các sản phẩm đều có điểm đề xuất từ cả hai phương pháp
            hybrid_products = pd.merge(
                hybrid_products,
                user_based[['product_id', 'recommendation_score']],
                on='product_id',
                how='left'
            )
            
            hybrid_products = pd.merge(
                hybrid_products,
                content_based[['product_id', 'similarity']],
                on='product_id',
                how='left'
            )
            
            # Điền các giá trị NaN với 0
            hybrid_products['recommendation_score'] = hybrid_products['recommendation_score'].fillna(0)
            hybrid_products['similarity'] = hybrid_products['similarity'].fillna(0)
            
            # Tính điểm kết hợp
            hybrid_products['hybrid_score'] = (
                hybrid_ratio * hybrid_products['recommendation_score'] + 
                (1 - hybrid_ratio) * hybrid_products['similarity']
            )
            
            # Lấy top 10 sản phẩm có điểm kết hợp cao nhất
            hybrid_products = hybrid_products.sort_values('hybrid_score', ascending=False).head(10)
            
            # Hiển thị tiêu đề 
            st.subheader(f"Sản phẩm đề xuất cho {customer_id2} kết hợp với mô tả của bạn:")
            
            # Hiển thị sản phẩm đề xuất
            display_product_cards(hybrid_products, score_col='hybrid_score')
            
            # Hiển thị phân tích
            st.subheader("Phân tích điểm đề xuất kết hợp")
            
            # Biểu đồ so sánh điểm đề xuất theo từng phương pháp
            comparison_df = pd.melt(
                hybrid_products[['product_name', 'recommendation_score', 'similarity', 'hybrid_score']],
                id_vars=['product_name'],
                value_vars=['recommendation_score', 'similarity', 'hybrid_score'],
                var_name='Score Type',
                value_name='Score Value'
            )
            
            # Đổi tên loại điểm để hiển thị dễ hiểu hơn
            comparison_df['Score Type'] = comparison_df['Score Type'].replace({
                'recommendation_score': 'Dựa trên người dùng',
                'similarity': 'Dựa trên nội dung',
                'hybrid_score': 'Kết hợp'
            })
            
            fig = px.bar(
                comparison_df,
                x='product_name',
                y='Score Value',
                color='Score Type',
                barmode='group',
                title="So sánh điểm đề xuất theo từng phương pháp",
                labels={'product_name': 'Sản phẩm', 'Score Value': 'Điểm đề xuất', 'Score Type': 'Phương pháp'}
            )
            fig.update_layout(xaxis_tickangle=-45)
            st.plotly_chart(fig, use_container_width=True)

# Footer
st.markdown("---")
st.markdown(
    """
    <div style="text-align: center;">
        <p>Hệ thống Đề xuất Sản phẩm Shopee | Dự án Machine Learning</p>
        <p>© 2025 - Phát triển Cao Thị Ngọc Minh & Nguyễn Kế Nhựt</p>
    </div>
    """,
    unsafe_allow_html=True
)