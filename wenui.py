import streamlit as st

def build_enhanced_interface():
    st.sidebar.title("高级选项")
    # 新增调试面板
    debug_mode = st.sidebar.checkbox("显示分析过程")
    retrieval_num = st.sidebar.slider("检索条款数", 3, 10, 5)
    # 主界面优化
    question = st.text_area("请输入法律问题：",
    placeholder="例如：试用期被辞退如何获得补偿？",
    height=120)
    if st.button("智能分析", help="点击提交法律咨询"):
        with st.spinner("正在检索相关法律条款..."):
            response = query_engine.query(
            question,
            retrieval_num=retrieval_num
            )
            # 主体识别高亮
            if "[用人单位]" in response.response:
                st.success("**主体类型识别**: 用人单位相关条款")
            elif "[劳动者]" in response.response:
                st.success("**主体类型识别**: 劳动者相关条款")
            # 分栏展示
            col1, col2 = st.columns([3, 2])
            with col1:
                st.markdown(f"### 法律建议\n{response.response}")
            with col2:
                with st.expander("⚖ 关联条款分析"):
                    visualize_score_distribution(response.scores)