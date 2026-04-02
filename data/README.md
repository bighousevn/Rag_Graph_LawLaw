[
    {
        "nodes": [
            {
                "id": "N01",
                "name": "Người lao động",
                "listSectionId": [
                    "1.1",
                    "1.2"
                ]
            },
            {
                "id": "N02",
                "name": "Hợp đồng lao động",
                "listSectionId": [
                    "1.1"
                ]
            },
            {
                "id": "N03",
                "name": "Tiền lương",
                "listSectionId": [
                    "1.2"
                ]
            }
        ],
        "relationships": [
            {
                "id": "E01",
                "name": "Ký kết",
                "source": "N01",
                "target": "N02",
                "listSectionId": [
                    "1.1"
                ]
            },
            {
                "id": "E02",
                "name": "Bao gồm",
                "source": "N02",
                "target": "N03",
                "listSectionId": [
                    "1.2"
                ]
            }
        ]
    }
] đây là data đầu vào, bạn hãy lấy các entity ra , mỗi entity tìm từ 0-5 từ đồng nghĩa , sau đó đưa từ đồng nghĩa sang định dạng như thế này[
    {
        "synonymId": "S01",
        "entityId": "N01",
        "entityName": "Người lao động",
        "synonym": "người làm việc"
    },
    {
        "synonymId": "S02",
        "entityId": "N02",
        "entityName": "Hợp đồng lao động",
        "synonym": "hợp đồng làm việc"
    }
] xuất json giúp tôi