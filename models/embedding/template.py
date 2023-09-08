from typing import Dict


class Template:
    
    def __init__(self, content:Dict[str, str]) -> None:
        self.content = content
        
    def _fit(self, content:Dict[str, str]):
        raise NotImplementedError

    def __str__(self) -> str:
        return self._fit(self.content)

class BasicTempalte(Template):
    
    def __init__(self, content:Dict[str, str]) -> None:
        super().__init__(content)
        self.output = self._fit(content)
    
    def __str__(self) -> str:
        return self.output

    def _fit(self, content: Dict[str, str]):
        content_list = []
        for key, val in content.items():
            content_list.append(self._template(key, val))
        return ",".join(content_list)
            
    def _template(self, subject, predicate) :
        return f"The {subject} is {predicate}"
