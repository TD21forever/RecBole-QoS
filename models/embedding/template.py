from typing import Dict


class Template:

    def __init__(self) -> None:
        ...
        
    def _fit(self):
        raise NotImplementedError


class BasicTempalte(Template):

    def __init__(self, content: Dict[str, str]) -> None:
        super().__init__()
        self.output = self._fit(content)

    def __str__(self) -> str:
        return self.output

    def _fit(self, content: Dict[str, str]):
        content_list = []
        for key, val in content.items():
            content_list.append(self._template(key, val))
        return ",".join(content_list)

    def _template(self, subject, predicate):
        return f"The {subject} is {predicate}"


class ImprovedTemplate(Template):

    def __init__(self, type, *args, **kwargs) -> None:
        super().__init__()
        if type == "user":
            self.output = self.fit_user(**kwargs)
        else:
            self.output = self.fit_service(**kwargs)

    def fit_user(self, *, user_id, ip_address, country, ip_number, AS, latitude, longitude, invocations):
        template = f"""
            The following content pertains to the personal attributes of the user {user_id}:
            When initiating requests, user {user_id} has the geographical coordinates {latitude} and {longitude}, located in {country}, belongs to autonomous system {AS}, and uses the IP address {ip_address} with the IP number {ip_number} while accessing web services.
            The following content pertains to the behavioral attributes of the user {user_id}:
            { "".join([self._user_invoke_service(user_id, service_id, qos)  for user_id, service_id, qos in invocations]) }
            If personal and behavioral attributes are similar, it is assumed that two users will have the same QoS values when invoking the same services.
        """
        return template

    def fit_service(self, *, service_id, wsdl_address, provider, ip_address, country, ip_number, AS, latitude, longitude, invocations):
        template = f"""
            The following content pertains to the personal attributes of the service {service_id}:
            The service {service_id} is provided by {provider} with wsdl_address {wsdl_address}, it has the geographical coordinates {latitude} and {longitude}, located in {country}, belongs to autonomous system {AS}, and uses the IP address {ip_address} with the IP number {ip_number}.
            The following content pertains to the behavioral attributes of the service {service_id}:
            { "".join([self._service_invoked_by_user(user_id, service_id, qos)  for user_id, service_id, qos in invocations]) }
            If personal and behavioral attributes are similar, it is assumed that two service will have the same QoS values when invoked by the same user.
        """
        return template

    def _user_invoke_service(self, user_id, service_id, qos):
        return f"User {user_id} invokes Service {service_id} with a QoS value of {qos}. \n"

    def _service_invoked_by_user(self, user_id, service_id, qos):
        return f"Service {service_id} is invoked by User {user_id} with a QoS value of {qos}. \n"

    def __str__(self) -> str:
        return self.output