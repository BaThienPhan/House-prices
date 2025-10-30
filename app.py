import streamlit as st
import joblib
import pandas as pd
import numpy as np

# THAY THẾ BẰNG TÊN FILE CỦA BẠN
# Lưu ý: joblib cần scikit-learn version 1.6.1 để tải file này
MODEL_FILENAME = "lasso_house_price_pipeline.joblib"

# --- 1. ĐỊNH NGHĨA TẤT CẢ FEATURES VÀ PHÂN LOẠI CỘT ---

# Định nghĩa tất cả các features mà mô hình cần (sau khi loại bỏ 'Id', 'Alley', 'PoolQC', 'Fence', 'MiscFeature', 'SalePrice')
ALL_FEATURES = [
    'MSSubClass', 'MSZoning', 'LotFrontage', 'LotArea', 'Street', 'LotShape', 'LandContour',
    'Utilities', 'LotConfig', 'LandSlope', 'Neighborhood', 'Condition1', 'Condition2',
    'BldgType', 'HouseStyle', 'OverallQual', 'OverallCond', 'YearBuilt', 'YearRemodAdd',
    'RoofStyle', 'RoofMatl', 'Exterior1st', 'Exterior2nd', 'MasVnrType', 'MasVnrArea',
    'ExterQual', 'ExterCond', 'Foundation', 'BsmtQual', 'BsmtCond', 'BsmtExposure',
    'BsmtFinType1', 'BsmtFinSF1', 'BsmtFinType2', 'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF',
    'Heating', 'HeatingQC', 'CentralAir', 'Electrical', '1stFlrSF', '2ndFlrSF',
    'LowQualFinSF', 'GrLivArea', 'BsmtFullBath', 'BsmtHalfBath', 'FullBath', 'HalfBath',
    'BedroomAbvGr', 'KitchenAbvGr', 'KitchenQual', 'TotRmsAbvGrd', 'Functional',
    'Fireplaces', 'FireplaceQu', 'GarageType', 'GarageYrBlt', 'GarageFinish', 'GarageCars',
    'GarageArea', 'GarageQual', 'GarageCond', 'PavedDrive', 'WoodDeckSF', 'OpenPorchSF',
    'EnclosedPorch', '3SsnPorch', 'ScreenPorch', 'PoolArea', 'MoSold', 'YrSold', 'SaleType',
    'SaleCondition', 'MiscVal'
]

# Định nghĩa các cột Phân loại (Categorical)
# >>> ĐÃ LOẠI BỎ 'MSSubClass' KHỎI DANH SÁCH NÀY <<<
cat_cols = [
    'MSZoning', 'Street', 'LotShape', 'LandContour', 'Utilities', 'LotConfig',
    'LandSlope', 'Neighborhood', 'Condition1', 'Condition2', 'BldgType', 'HouseStyle',
    'RoofStyle', 'RoofMatl', 'Exterior1st', 'Exterior2nd', 'MasVnrType', 'ExterQual',
    'ExterCond', 'Foundation', 'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1',
    'BsmtFinType2', 'Heating', 'HeatingQC', 'CentralAir', 'Electrical', 'KitchenQual',
    'Functional', 'FireplaceQu', 'GarageType', 'GarageFinish', 'GarageQual', 'GarageCond',
    'PavedDrive', 'SaleType', 'SaleCondition'
]

# Định nghĩa các cột Số (Numerical)
# >>> ĐÃ THÊM 'MSSubClass' VÀO DANH SÁCH NÀY <<<
num_cols = [
    'MSSubClass',  # <-- Đã chuyển từ cat_cols sang
    'LotFrontage', 'LotArea', 'OverallQual', 'OverallCond', 'YearBuilt', 'YearRemodAdd',
    'MasVnrArea', 'BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF', '1stFlrSF',
    '2ndFlrSF', 'LowQualFinSF', 'GrLivArea', 'BsmtFullBath', 'BsmtHalfBath', 'FullBath',
    'HalfBath', 'BedroomAbvGr', 'KitchenAbvGr', 'TotRmsAbvGrd', 'Fireplaces', 'GarageYrBlt',
    'GarageCars', 'GarageArea', 'WoodDeckSF', 'OpenPorchSF', 'EnclosedPorch', '3SsnPorch',
    'ScreenPorch', 'PoolArea', 'MoSold', 'YrSold',
    'MiscVal'
]

# --- 2. TẢI PIPELINE VÀ DỮ LIỆU CẦN THIẾT ---
try:
    pipeline = joblib.load(MODEL_FILENAME)
except FileNotFoundError:
    st.error(
        f"Lỗi: Không tìm thấy file mô hình '{MODEL_FILENAME}'. Vui lòng kiểm tra lại đường dẫn.")
    st.stop()
except AttributeError as e:
    st.error(f"Lỗi Tải Mô hình (AttributeError): {e}")
    st.warning("Có vẻ như phiên bản scikit-learn dùng để lưu mô hình (Colab: 1.6.1) không khớp với phiên bản hiện tại của bạn.")
    st.info("Vui lòng chạy lệnh sau trong terminal của bạn để khắc phục:")
    st.code("pip uninstall scikit-learn\npip install scikit-learn==1.6.1")
    st.stop()


# --- 3. XÂY DỰNG GIAO DIỆN STREAMLIT ---
st.set_page_config(page_title="Dự đoán Giá nhà", layout="wide")
st.title("🏡 Ứng dụng Dự đoán Giá nhà (Mô hình Ridge/Lasso)")
st.write("Vui lòng nhập các thông số chính của căn nhà để nhận dự đoán giá.")

# Chia layout thành cột để dễ nhập liệu hơn
col1, col2, col3 = st.columns(3)
input_data = {}

# Danh sách các lựa chọn (Chỉ là ví dụ, bạn nên dùng TẤT CẢ các giá trị có trong data train)
kitchen_qual_options = ('Ex', 'Gd', 'TA', 'Fa', 'Po')
neighborhood_options = ('CollgCr', 'Veenker', 'Mitchel', 'NoRidge', 'NWAmes', 'Somerst', 'OldTown', 'BrkSide', 'Sawyer',
                        'NridgHt', 'NAmes', 'SawyerW', 'IDOTRR', 'MeadowV', 'Edwards', 'Timber', 'Gilbert',
                        'StoneBr', 'ClearCr', 'NPkVill', 'Blmngtn', 'BrDale', 'SWISU', 'Blueste')


with col1:
    st.header("Thông số Chính")
    # Các input quan trọng
    input_data['OverallQual'] = st.slider(
        '1. Chất lượng tổng thể (OverallQual) [1-10]', 1, 10, 7)
    input_data['GrLivArea'] = st.number_input(
        '2. Diện tích Sinh hoạt (GrLivArea, SqFt)', min_value=300, max_value=5000, value=1800, step=10)
    input_data['YearBuilt'] = st.slider(
        '3. Năm Xây dựng (YearBuilt)', 1800, 2025, 2005)

with col2:
    st.header("Thông số Garage & Bếp")
    input_data['GarageCars'] = st.slider(
        '4. Sức chứa Garage (GarageCars)', 0, 4, 2)
    input_data['FullBath'] = st.slider(
        '5. Số phòng tắm đầy đủ (FullBath)', 0, 4, 2)
    input_data['KitchenQual'] = st.selectbox(
        '6. Chất lượng Bếp (KitchenQual)', kitchen_qual_options, index=kitchen_qual_options.index('Gd'))

with col3:
    st.header("Thông số Vị trí")
    input_data['Neighborhood'] = st.selectbox(
        '7. Khu vực (Neighborhood)', neighborhood_options, index=neighborhood_options.index('NridgHt'))
    input_data['LotArea'] = st.number_input(
        '8. Diện tích Lô đất (LotArea, SqFt)', min_value=1000, max_value=50000, value=10000, step=100)
    # Thêm một cột quan trọng khác
    input_data['TotalBsmtSF'] = st.number_input(
        '9. Tổng diện tích tầng hầm (TotalBsmtSF, SqFt)', min_value=0, max_value=3000, value=1000, step=10)


# --- 4. TẠO DATAFRAME VÀ DỰ ĐOÁN ---
if st.button('DỰ ĐOÁN GIÁ NHÀ', type="primary"):

    # Chuẩn bị input thô: tạo một DataFrame với TẤT CẢ các cột mô hình yêu cầu
    final_input = {}

    # 1. Điền các giá trị người dùng đã nhập
    for col, value in input_data.items():
        final_input[col] = value

    # 2. Điền các giá trị mặc định cho các cột không có trên UI (Tất cả các cột phải tồn tại)
    for feature in ALL_FEATURES:
        if feature not in final_input:
            # Gán giá trị mặc định dựa trên loại cột
            if feature in cat_cols:
                # Giá trị mặc định phổ biến cho cột phân loại
                if 'Qual' in feature or 'Cond' in feature:
                    final_input[feature] = 'TA'
                elif feature in ['MasVnrType', 'FireplaceQu', 'GarageType', 'GarageFinish', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2']:
                    final_input[feature] = 'None'
                else:
                    # Giá trị an toàn cho các cột phân loại còn lại
                    final_input[feature] = 'Other'

            elif feature in num_cols:
                # Giá trị mặc định an toàn cho cột số là 0
                final_input[feature] = 0.0

            else:
                # Trường hợp còn lại (nếu có lỗi trong danh sách cat/num)
                st.warning(
                    f"Cảnh báo: Cột '{feature}' không được gán giá trị mặc định. Đặt bằng 0.")
                final_input[feature] = 0.0

    # Tạo DataFrame cho dự đoán (Quan trọng: Phải theo thứ tự và tên cột chính xác)
    # Sắp xếp lại final_input theo thứ tự ALL_FEATURES để tránh lỗi trong Pipeline
    ordered_input = {k: final_input[k] for k in ALL_FEATURES}
    input_df = pd.DataFrame([ordered_input])

    # Thực hiện dự đoán bằng Pipeline
    try:
        # Pipeline thường mong đợi input là DataFrame
        predicted_price_log = pipeline.predict(input_df)[0]
        # Chuyển đổi từ log(giá) sang giá trị thực (exp là e mũ)
        predicted_price = np.expm1(predicted_price_log)

        st.subheader("💰 Kết quả Dự đoán Giá nhà:")
        st.balloons()
        st.metric(label="Giá nhà dự đoán (USD)",
                  value=f"${predicted_price:,.0f}")
        st.markdown(f"""
        <div style='background-color: #e6f7ff; padding: 15px; border-radius: 10px;'>
            <p style='font-size: 16px; margin: 0;'>
                <strong>Lưu ý:</strong> Kết quả này là ước tính. Mô hình đã dự đoán giá trị là 
                <span style='color: #0077cc;'>log(Giá) = {predicted_price_log:.4f}</span>. 
                Giá thực tế có thể thay đổi tùy thuộc vào thị trường.
            </p>
        </div>
        """, unsafe_allow_html=True)

    except Exception as e:
        st.error(f"Đã xảy ra lỗi trong quá trình dự đoán: {e}")
        st.warning(
            "Vui lòng kiểm tra lại các giá trị nhập hoặc kiểm tra Pipeline của bạn.")
        st.info(
            "Kiểm tra lỗi tiềm ẩn: Tên cột không khớp, hoặc Pipeline đang mong đợi thứ tự/loại dữ liệu khác.")
        # Hiển thị các cột trong input_df để debug
        st.caption("Các cột được cung cấp cho Pipeline:")
        st.dataframe(pd.DataFrame(input_df.columns))

# --- 5. HƯỚNG DẪN BỔ SUNG (Giúp người dùng khắc phục lỗi phiên bản) ---
st.sidebar.markdown("---")
st.sidebar.markdown("## 🔧 Hướng dẫn Khắc phục Lỗi")
st.sidebar.info(
    "Nếu bạn gặp lại lỗi `AttributeError` (ví dụ: `_RemainderColsList`), đó là do xung đột phiên bản scikit-learn. "
    "Mô hình được lưu với **scikit-learn 1.6.1**, nhưng máy của bạn đang dùng phiên bản khác. "
    "Vui lòng đồng bộ hóa bằng cách chạy lệnh sau trong terminal:"
)
st.sidebar.code("pip uninstall scikit-learn\npip install scikit-learn==1.6.1")
