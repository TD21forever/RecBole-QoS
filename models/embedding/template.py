from typing import Dict


class Template:

    def __init__(self) -> None:
        ...
        
    def _fit(self):
        raise NotImplementedError


class BasicTempalte(Template):

    def __init__(self, content: Dict[str, str]) -> None:
        self.output = self._fit(content)
        super().__init__()
        

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

    def __init__(self, type, content, invocations) -> None:
        super().__init__()
        if type == "user":
            self.output = self.fit_user(invocations=invocations, **content)
        else:
            self.output = self.fit_service(invocations=invocations, **content)

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

    def template(self):
        """
        The following content pertains to the personal attributes of the user 0:

        When initiating requests, user 0 has the geographical coordinates 38.0 and -97.0, 
        located in United States, belongs to autonomous system AS7018 AT&T Services, Inc., 
        and uses the IP address 12.108.127.138 with the IP number 208437130 while accessing web services.

        The following content pertains to the behavioral attributes of the user 0:

        User 0.0 invokes Service 3650.0 with a QoS value of 0.08900000154972076. 
        User 0.0 invokes Service 1551.0 with a QoS value of 0.5709999799728394. 
        User 0.0 invokes Service 1272.0 with a QoS value of 0.1420000046491623. 
        User 0.0 invokes Service 2328.0 with a QoS value of 6.758999824523926. 
        ….
        If personal and behavioral attributes are similar, it is assumed that two users will have the same QoS values when invoking the same services.
 
        """