import streamlit as st
import datetime
import pandas as pd
from utils.google_sheets_loader import append_row_to_sheet

def render_data_entry_tab():
    st.header("📤 Submit New Shipment Record")

    st.subheader("📝 Fill Individual Entry")
    with st.form("entry_form"):
        col1, col2 = st.columns(2)
        product_group = col1.text_input("Product Group")
        vendor = col2.text_input("Vendor")

        col3, col4 = st.columns(2)
        site = col3.text_input("Manufacturing Site")
        country = col4.text_input("Country")

        shipment_mode = st.selectbox("Shipment Mode", ["Air", "Air charter", "Ocean", "Truck", "Unknown"])

        col5, col6 = st.columns(2)
        quantity = col5.number_input("Line Item Quantity", min_value=0, step=1)
        unit_price = col6.number_input("Unit Price (USD)", min_value=0.0, format="%.2f")

        col7, col8 = st.columns(2)
        freight_cost = col7.number_input("Freight Cost (USD)", min_value=0.0, format="%.2f")
        scheduled_date = col8.date_input("Scheduled Delivery Date", value=datetime.date.today())

        delivered_date = st.date_input("Delivered to Client Date", value=datetime.date.today())

        col9, col10 = st.columns(2)
        record_id = col9.text_input("ID")
        asn_dn = col10.text_input("ASN/DN #")

        col11, col12 = st.columns(2)
        dosage_form = col11.text_input("Dosage Form")
        sub_class = col12.text_input("Sub Classification")

        col13, col14 = st.columns(2)
        po_so = col13.text_input("PO / SO #")
        pq = col14.text_input("PQ #")

        project_code = st.text_input("Project Code")

        col15, col16 = st.columns(2)
        pq_sent_date = col15.date_input("PQ First Sent to Client Date", value=datetime.date.today())
        po_sent_date = col16.date_input("PO Sent to Vendor Date", value=datetime.date.today())

        delivery_recorded_date = st.date_input("Delivery Recorded Date", value=datetime.date.today())

        col17, col18 = st.columns(2)
        weight = col17.number_input("Weight (Kilograms)", min_value=0.0, format="%.2f")
        line_item_value = col18.number_input("Line Item Value", min_value=0.0, format="%.2f")

        col19, col20 = st.columns(2)
        uom = col19.text_input("Unit of Measure (UOM)")
        funding_source = col20.text_input("Funding Source")

        col21, col22 = st.columns(2)
        batch_number = col21.text_input("Batch Number")
        expiry_date = col22.date_input("Expiry Date", value=datetime.date.today())

        col23, col24 = st.columns(2)
        manufacturer = col23.text_input("Manufacturer")
        product_description = col24.text_input("Product Description")

        col25, col26 = st.columns(2)
        delivery_note = col25.text_input("Delivery Note Number")
        document_id = col26.text_input("Document ID")

        col27, col28 = st.columns(2)
        requisition_id = col27.text_input("Requisition ID")
        facility_name = col28.text_input("Facility Name")

        facility_code = st.text_input("Facility Code")

        submitted = st.form_submit_button("Submit")

        if submitted:
            new_row = [
                product_group, vendor, site, country, shipment_mode,
                quantity, unit_price, freight_cost,
                scheduled_date.strftime("%Y-%m-%d"),
                delivered_date.strftime("%Y-%m-%d"),
                record_id, asn_dn, dosage_form, sub_class,
                po_so, pq, project_code,
                pq_sent_date.strftime("%Y-%m-%d"), po_sent_date.strftime("%Y-%m-%d"),
                delivery_recorded_date.strftime("%Y-%m-%d"), weight, line_item_value,
                uom, funding_source, batch_number,
                expiry_date.strftime("%Y-%m-%d"), manufacturer, product_description,
                delivery_note, document_id, requisition_id, facility_name, facility_code
            ]

            success = append_row_to_sheet(new_row)

            if success:
                st.success("✅ Record submitted successfully!")
            else:
                st.error("❌ Submission failed. Please check the logs or try again.")

    st.divider()

    st.subheader("📁 Upload Bulk Records via CSV")
    uploaded_file = st.file_uploader("Upload CSV", type=["csv"])

    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            st.dataframe(df.head())
            if st.button("Submit CSV Records"):
                all_success = True
                for _, row in df.iterrows():
                    row_data = row.fillna("").tolist()
                    success = append_row_to_sheet(row_data)
                    if not success:
                        all_success = False
                if all_success:
                    st.success("✅ All CSV records submitted successfully!")
                else:
                    st.warning("⚠️ Some rows failed to submit. Check your data.")
        except Exception as e:
            st.error(f"❌ Error processing CSV: {e}")