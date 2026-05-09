
## Ghi chú xử lý dữ liệu

### YenTran
- **Tiki**: Đã xóa duplicate product_id trước khi push
- **Lazada**: Đã lọc title spam CAPS (title có >50% từ viết hoa toàn bộ) trước khi push

### YenNguyen
- **Lazada**: Một số branch có cấu trúc thư mục chưa đúng chuẩn (CSV nằm ở root thay vì data/raw/), cần kiểm tra và fix lại

## Thống kê dataset sau merge

- Tổng số records: ~20,000 SP
- Nguồn: Tiki (L2) + Lazada (L3)
- File merge: `processed-data/all_platforms_merged.csv`
- File report: `processed-data/statistics_combined_20260509_201106.csv`

### Fill rate tổng quan
- price: 100%
- brand: ~30-100% (thấp hơn ở Lazada do nhiều hàng no-brand)
- rating/review: ~50-90% (sản phẩm mới chưa có đánh giá là bình thường)
