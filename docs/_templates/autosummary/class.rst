{{ fullname | escape | underline}}

.. currentmodule:: {{ module }}

.. autoclass:: {{ objname }}

{% block methods %}
{% if methods %}
.. rubric:: Methods

.. autosummary::
   :toctree:
   :nosignatures:
{% for item in methods %}
   {%- if item not in ['__init__', 'format', 'format_map'] %}
   ~{{ name }}.{{ item }}
   {%- endif %}
{%- endfor %}
{% endif %}
{% endblock %}
